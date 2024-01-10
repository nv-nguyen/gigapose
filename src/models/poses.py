import torch
from einops import einsum, rearrange, repeat
from megapose.utils.logging import get_logger
from src.lib3d.torch import inverse_affine, normalize_affine_transform
from src.models.ransac import RANSAC
import src.megapose.utils.tensor_collection as tc
import pandas as pd

logger = get_logger(__name__)


class ObjectPoseRecovery(torch.nn.Module):
    def __init__(
        self,
        template_K,
        template_Ms,
        template_poses,
        pixel_threshold=14,
    ):
        super(ObjectPoseRecovery, self).__init__()
        self.template_K = template_K
        self.template_Ms = template_Ms
        self.template_poses = template_poses
        self.ransac = RANSAC(pixel_threshold=pixel_threshold)

    def _forward_recovery(
        self,
        query_M,
        query_K,
        pred_view_ids,
        pred_Ms,
        template_K,
        template_Ms,
        template_poses,
    ):
        """
        Recover 6D pose from 2D predictions
        1. Rotation = Inplane (Kabsch) rotation + Viewpoint rotation
        2. 2D translation = 2D (Kabsch) transform + Crop transform of query and template
        3. Z (on the ray) scale = 2D scale + Focal length ratio
        Input:
            query_M: (B, 3, 3)
            query_K: (B, 3, 3)

            pred_view_ids: (B, N)
            pred_Ms: (B, N, 3, 3)

            template_K: (B, 3, 3)
            template_Ms: (B, N, 3, 3)
            template_poses: (B, N, 4, 4)
        """
        B, N = pred_view_ids.shape[:2]
        pred_view_ids_3x3 = repeat(pred_view_ids, "b n -> b n 3 3")
        pred_view_ids_4x4 = repeat(pred_view_ids, "b n -> b n 4 4")

        temp_Ks = repeat(template_K, "b h w -> b n h w", n=N)
        temp_Ms = torch.gather(
            template_Ms,
            1,
            pred_view_ids_3x3,
        )

        # Step 1: Rotation = Azmith, Elevation + Inplane (Kabsch) rotation
        pred_poses = torch.gather(
            template_poses,
            1,
            pred_view_ids_4x4,
        )
        pred_R_inplane = normalize_affine_transform(pred_Ms)

        pred_poses[:, :, :3, :3] = torch.matmul(
            pred_R_inplane, pred_poses[:, :, :3, :3]
        )

        # Step 2: 2D translation
        temp_z = pred_poses[:, :, 2, 3].clone()
        temp_translation = rearrange(pred_poses[:, :, :3, 3], "b n c -> b n c 1")
        temp_center2d = torch.matmul(temp_Ks, temp_translation)
        temp_center2d = temp_center2d / temp_center2d[:, :, 2].unsqueeze(2)

        # fully 2D affine transform from template to query
        inv_query_M = inverse_affine(query_M)
        inv_query_M = inv_query_M.unsqueeze(1).repeat(1, N, 1, 1)
        affine2d = torch.matmul(torch.matmul(inv_query_M, pred_Ms), temp_Ms)

        # recover 2D center of query
        query_center2d = torch.matmul(affine2d, temp_center2d)
        query_Ks = repeat(query_K, "b h w -> b n h w", n=N)
        inv_query_Ks = torch.inverse(query_Ks)

        # recover Z_query = (Z_temp / scale_2d) * (focal length ratio)
        scale2d = torch.norm(affine2d[:, :, :2, 0], dim=2)
        focal_ratio = query_Ks[:, :, 0, 0] / temp_Ks[:, :, 0, 0]
        query_z = (temp_z / scale2d) * focal_ratio

        # combine 2D translation and Z scale
        query_translation = torch.matmul(inv_query_Ks, query_center2d).squeeze(-1)
        query_translation /= query_translation[:, :, 2].unsqueeze(-1)
        pred_poses[:, :, :3, 3] = query_translation * query_z.unsqueeze(-1)

        return pred_poses

    def forward_recovery(
        self,
        tar_label,
        tar_K,
        tar_M,
        pred_src_views,
        pred_M,
    ):
        template_poses = self.template_poses.clone()[tar_label - 1]
        template_Ms = self.template_Ms.clone()[tar_label - 1]
        template_K = self.template_K.clone()[tar_label - 1]
        return self._forward_recovery(
            query_M=tar_M,
            query_K=tar_K,
            pred_view_ids=pred_src_views,
            pred_Ms=pred_M,
            template_K=template_K,
            template_Ms=template_Ms,
            template_poses=template_poses,
        )

    def forward_ransac(
        self,
        predictions,
    ):
        batch_size, k = predictions.src_pts.shape[:2]
        device = predictions.src_pts.device

        pred_Ms = torch.eye(3).to(device).unsqueeze(0).repeat(batch_size, k, 1, 1)
        idx_faileds = torch.zeros(batch_size, k, device=device).bool()

        tmp = []
        for idx_k in range(k):
            small_batch = tc.PandasTensorCollection(
                src_pts=predictions.src_pts[:, idx_k],
                tar_pts=predictions.tar_pts[:, idx_k],
                relScale=predictions.relScale[:, idx_k],
                relInplane=predictions.relInplane[:, idx_k],
                infos=pd.DataFrame(),
            )
            Ms, idx_failed, out_data = self.ransac(small_batch)
            pred_Ms[:, idx_k], idx_faileds[:, idx_k] = Ms, idx_failed
            # sort by num_inliers
            tmp.append(out_data)

        # sort by num_inliers
        inlier_src_pts, inlier_tar_pts, inlier_scores = [], [], []
        for idx_k in range(k):
            inlier_src_pts.append(tmp[idx_k].src_pts)
            inlier_tar_pts.append(tmp[idx_k].tar_pts)
            inlier_scores.append(tmp[idx_k].scores)
        inlier_src_pts = torch.stack(inlier_src_pts, dim=1)
        inlier_tar_pts = torch.stack(inlier_tar_pts, dim=1)
        inlier_scores = torch.stack(inlier_scores, dim=1)

        predictions.register_tensor("idx_failed", idx_faileds)
        predictions.register_tensor("M", pred_Ms)
        predictions.register_tensor("ransac_scores", inlier_scores)
        predictions.register_tensor("ransac_src_pts", inlier_src_pts)
        predictions.register_tensor("ransac_tar_pts", inlier_tar_pts)
        return predictions
