import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import src.megapose.utils.tensor_collection as tc
from src.utils.batch import BatchedData


class LocalSimilarity(torch.nn.Module):
    def __init__(
        self,
        k,
        sim_threshold,
        patch_threshold,
        search_direction="tar2src",
        image_size=224,
        patch_size=14,
        max_batch_size=32,
    ) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.k = k
        self.sim_threshold = sim_threshold
        self.patch_threshold = patch_threshold
        self.search_direction = search_direction
        self.num_patches = image_size // patch_size
        self.idx_gt = torch.arange(0, self.num_patches * self.num_patches)

    def format_prediction(self, src_mask, input_tar_pts):
        """
        Formatting predictions by assign -1 to outside of src_mask and convert to (B, N, (H W), 2) format
        """
        if len(src_mask.shape) == 3:
            src_mask = src_mask.unsqueeze(1)
            input_tar_pts = input_tar_pts.unsqueeze(1)
            is_3dim = True
        else:
            is_3dim = False
        B, N, H, W = src_mask.shape
        device = src_mask.device

        src_pts_ = torch.nonzero(src_mask)
        b, n, h, w = (
            src_pts_[:, 0],
            src_pts_[:, 1],
            src_pts_[:, 2],
            src_pts_[:, 3],
        )
        src_pts = torch.full((B, N, H, W, 2), -1, dtype=torch.long, device=device)
        tar_pts = torch.full((B, N, H, W, 2), -1, dtype=torch.long, device=device)

        src_pts[b, n, h, w] = src_pts_[:, [3, 2]]  # swap x, y
        tar_pts[b, n, h, w] = input_tar_pts[b, n, h, w]

        src_pts = rearrange(src_pts, "b n h w c -> b n (h w) c")
        tar_pts = rearrange(tar_pts, "b n h w c -> b n (h w) c")

        if is_3dim:
            src_pts = src_pts.squeeze(1)
            tar_pts = tar_pts.squeeze(1)
        return src_pts, tar_pts

    def convert_index2location(self, index):
        """Convert from (H*W) index to (H, W) location"""
        h = index // self.num_patches
        w = index % self.num_patches
        patch_location = torch.stack([w, h], dim=-1)
        return patch_location.float()

    def convert_location2index(self, location):
        """Convert from (H, W) location to (H*W) index"""
        if len(location.shape) == 2:
            index = location[:, 1] * self.num_patches + location[:, 0]
        elif len(location.shape) == 3:
            index = location[:, :, 1] * self.num_patches + location[:, :, 0]
        else:
            raise ValueError("location must be 2D or 3D")
        return index.long()

    def find_consistency_patches(self, sim_src2tar, idx_src2tar, idx_tar2src):
        """Find the consistency patches between source and target image nearest neighbor search"""

        # cycle consistency (source -> target -> source)
        if len(idx_src2tar.shape) == 2:
            sim_src2tar = sim_src2tar.unsqueeze(1)
            idx_src2tar = idx_src2tar.unsqueeze(1)
            idx_tar2src = idx_tar2src.unsqueeze(1)
            is_2dim = True
        else:
            is_2dim = False
        B, N, Q = idx_src2tar.shape
        idx_gt = repeat(self.idx_gt.clone(), "m -> b n m", b=B, n=N)

        # compute the distance to find the consistency patches
        idx_src2src = torch.gather(idx_src2tar, 2, idx_tar2src)

        idx_src2src_2d = self.convert_index2location(idx_src2src)
        idx_gt_2d = self.convert_index2location(idx_gt)
        idx_gt_2d = idx_gt_2d.to(idx_src2src.device)
        distance = torch.norm(
            idx_src2src_2d - idx_gt_2d,
            dim=3,
        )  # b (x n) x q
        mask_dist = distance <= self.patch_threshold

        # compute the similarity to find the consistency patches (source -> target -> source)
        sim_src2src = torch.gather(sim_src2tar, 2, idx_tar2src)
        mask_sim = sim_src2src >= self.sim_threshold

        if is_2dim:
            mask_dist = mask_dist.squeeze(1)
            mask_sim = mask_sim.squeeze(1)
        return torch.logical_and(mask_dist, mask_sim)

    def val(
        self,
        src_feat,
        tar_feat,
        src_mask,
        tar_mask,
    ):
        """
        Find the nearest neighbor in the reference image for each patch in the query image
        src_feat: (B, C, H, W)  # best template
        tar_feat: (B, C, H, W) # real image
        """
        B, C, h, w = src_feat.shape
        feat_size = (self.num_patches, self.num_patches)

        tar_mask = F.interpolate(tar_mask.unsqueeze(1), size=feat_size)
        tar_mask = rearrange(tar_mask, "b 1 h w -> b (h w)")
        tar_feat = F.normalize(tar_feat, dim=1)
        tar_feat = rearrange(tar_feat, "b c h w -> b c (h w)")

        src_mask = F.interpolate(src_mask.unsqueeze(1), size=feat_size)
        src_mask = rearrange(src_mask, "b 1 H W -> b (H W)")
        src_feat = F.normalize(src_feat, dim=1)
        src_feat = rearrange(src_feat, "b c H W -> b c (H W)")

        # Step 1: Find nearest neighbor for each patch of query image
        sim = torch.einsum("b c t, b c s -> b t s", tar_feat, src_feat)
        sim *= src_mask[:, None, :]
        sim *= tar_mask[:, :, None]
        sim[sim < self.sim_threshold] = 0

        # Find nearest neighbor for each patch of query image
        score_tar2src, idx_tar2src = torch.max(sim, dim=2)  # b x t
        score_src2tar, idx_src2tar = torch.max(sim, dim=1)  # b x s
        mask_sim = score_tar2src != 0

        # Find consistency patches (source -> target -> source)
        if self.patch_threshold > 0:
            mask_cycle = self.find_consistency_patches(
                sim_src2tar=score_src2tar,
                idx_src2tar=idx_src2tar,
                idx_tar2src=idx_tar2src,
            )
        else:
            raise ValueError("patch_threshold must be greater than 0")

        mask_tar2src = torch.gather(src_mask, 1, idx_tar2src)
        mask_non_zero = (
            tar_mask  # mask of query = 0
            * mask_tar2src  # mask of template = 0
            * (idx_src2tar != 0)  # sim = 0
            * (idx_tar2src != 0)  # sim = 0
        )

        # Combine all masks
        t_mask = mask_sim * mask_cycle * mask_non_zero  # b x t
        t_mask = rearrange(t_mask, "b (h w) -> b h w", h=h)

        pred_src_pts = self.convert_index2location(idx_tar2src)
        pred_src_pts = rearrange(pred_src_pts, "b (h w) m -> b h w m", h=h).long()

        # Step 4: format the output (-1 as invalid)
        pred_tar_pts, pred_src_pts = self.format_prediction(
            src_mask=t_mask, input_tar_pts=pred_src_pts.long()
        )
        out_data = tc.PandasTensorCollection(
            infos=pd.DataFrame(),
            src_pts=pred_src_pts,
            tar_pts=pred_tar_pts,
            score=score_tar2src,
        )
        return out_data

    def test(
        self,
        src_feats,
        tar_feat,
        src_masks,
        tar_mask,
        max_batch_size=None,
    ):
        """
        Find the nearest neighbor in the reference image for each patch in the query image
        src_feat: (B, N, C, H, W)  # template
        tar_feat: (B, C, H, W) # real image
        """
        if max_batch_size is None:
            max_batch_size = self.max_batch_size
        in_names = ["src_feats", "tar_feat", "src_masks", "tar_mask"]
        in_data = [src_feats, tar_feat, src_masks, tar_mask]
        in_data = {
            name: BatchedData(batch_size=max_batch_size, data=data)
            for name, data in zip(in_names, in_data)
        }
        out_names = ["id_src", "score_src", "score_pts", "tar_pts", "src_pts"]
        out_data = {name: BatchedData(None) for name in out_names}

        for idx in range(len(in_data["src_feats"])):
            src_feats = in_data["src_feats"][idx]
            tar_feat = in_data["tar_feat"][idx]
            src_masks = in_data["src_masks"][idx]
            tar_mask = in_data["tar_mask"][idx]

            B, N = src_masks.shape[:2]
            device = src_masks.device
            feat_size = (self.num_patches, self.num_patches)

            tar_mask = F.interpolate(tar_mask.unsqueeze(1), size=feat_size)
            tar_mask = rearrange(tar_mask, "b 1 h w -> b (h w)")
            tar_feat = F.normalize(tar_feat, dim=1)
            tar_feat = rearrange(tar_feat, "b c h w -> b c (h w)")

            src_masks = F.interpolate(src_masks, size=feat_size)
            src_masks = rearrange(src_masks, "b n h w -> b n (h w)")
            src_feats = F.normalize(src_feats, dim=2)
            src_feats = rearrange(src_feats, "b n c h w -> b n c (h w)")

            # Step 1: Find nearest neighbor for each patch of query image
            sim = torch.einsum("b c t, b n c s -> b n t s", tar_feat, src_feats)
            sim *= src_masks[:, :, None, :]
            sim *= tar_mask[:, None, :, None]
            sim[sim < self.sim_threshold] = 0

            # Find nearest neighbor for each patch of query image
            if self.search_direction == "tar2src":
                score_tar2src, idx_tar2src = torch.max(sim, dim=3)  # b x n x t
                score_src2tar, idx_src2tar = torch.max(sim, dim=2)  # b x n x s
            elif self.search_direction == "src2tar":
                score_tar2src, idx_tar2src = torch.max(sim, dim=2)  # b x n x s
                score_src2tar, idx_src2tar = torch.max(sim, dim=3)  # b x n x t

            # Filter out the slow score matching
            mask_sim = score_tar2src >= self.sim_threshold

            # Find consistency patches (source -> target -> source)
            if self.patch_threshold > 0:
                mask_cycle = self.find_consistency_patches(
                    sim_src2tar=score_src2tar,
                    idx_src2tar=idx_src2tar,
                    idx_tar2src=idx_tar2src,
                )
            else:
                mask_cycle = torch.ones_like(mask_sim)

            # Find valid patches has mask in both source and target
            tar_masks = repeat(tar_mask, "b t -> b n t", n=N)
            mask_tar2src = torch.gather(src_masks, 2, idx_tar2src)

            mask_non_zero = (
                tar_masks  # mask of query = 0
                * mask_tar2src  # mask of template = 0
                * (idx_src2tar != 0)  # sim = 0
                * (idx_tar2src != 0)  # sim = 0
            )

            # Combine all masks
            mask_all = mask_sim * mask_cycle * mask_non_zero  # b x t

            # Step 2: Find best template for each target
            mask = mask_all.sum(dim=2) > 0
            sim_avg = torch.zeros(B, N, device=device)
            sim_avg[mask] = torch.sum(score_tar2src * mask_all, dim=2)[mask] / (
                self.num_patches**2
            )
            pred_score_src, pred_id_src = torch.topk(sim_avg, self.k, dim=1)

            # Step 3: Save the results
            idx_sample = torch.arange(0, B).to(device)
            idx_sample = repeat(idx_sample, "b -> b k", k=self.k)

            pred_tar_mask = mask_all[idx_sample, pred_id_src, :]  # b x k x t
            pred_score_pts = score_tar2src[idx_sample, pred_id_src, :]  # b x k x t
            pred_src_pts = self.convert_index2location(
                idx_tar2src[idx_sample, pred_id_src, :]
            )
            pred_tar_mask = rearrange(
                pred_tar_mask, "b k (h w) -> b k h w", h=self.num_patches
            )
            pred_src_pts = rearrange(
                pred_src_pts, "b k (h w) m -> b k h w m", h=self.num_patches
            )

            # Step 4: format the output (-1 as invalid)
            pred_tar_pts, pred_src_pts = self.format_prediction(
                src_mask=pred_tar_mask, input_tar_pts=pred_src_pts.long()
            )

            out_data["id_src"].cat(pred_id_src)
            out_data["score_src"].cat(pred_score_src)
            out_data["score_pts"].cat(pred_score_pts)
            out_data["tar_pts"].cat(pred_tar_pts)
            out_data["src_pts"].cat(pred_src_pts)

        out_data = tc.PandasTensorCollection(
            infos=pd.DataFrame(),
            id_src=out_data["id_src"].data,
            score_src=out_data["score_src"].data,
            score_pts=out_data["score_pts"].data,
            tar_pts=out_data["tar_pts"].data,
            src_pts=out_data["src_pts"].data,
        )
        return out_data
