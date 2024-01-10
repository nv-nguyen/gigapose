from dataclasses import dataclass
import torch
from src.utils.logging import get_logger
from src.utils.inout import MAX_VALUES
from src.lib3d.torch import (
    project_points,
    unproject_points,
    homogenuous,
    inverse_affine,
)

logger = get_logger(__name__)


@dataclass
class KeypointInput:
    def __init__(self, K, full_depth, mask, M, full_rgb=None, rgb=None):
        self.K = K
        self.full_depth = full_depth
        self.M = M
        self.mask = mask
        self.rgb = rgb
        self.full_rgb = full_rgb


@dataclass
class Keypoint:
    def __init__(self, src=None, tar=None, src_mask=None, tar_mask=None):
        self.src = src
        self.tar = tar
        B, N = src.shape[:2]
        device = src.device
        if src_mask is None:
            self.src_mask = torch.ones(B, N, device=device).bool()
            self.tar_mask = torch.ones(B, N, device=device).bool()
        else:
            self.src_mask = src_mask
            self.tar_mask = tar_mask

    def clone(self):
        return Keypoint(
            src=self.src.clone(),
            tar=self.tar.clone(),
            src_mask=self.src_mask.clone(),
            tar_mask=self.tar_mask.clone(),
        )

    def mask(self, name, mask):
        points = getattr(self, name).long()
        point_masks = getattr(self, f"{name}_mask")

        idx = torch.arange(points.shape[0])
        idx = idx.unsqueeze(-1).repeat(1, points.shape[1])

        outside_left = torch.logical_or(points[:, :, 0] < 0, points[:, :, 1] < 0)
        outside_right = torch.logical_or(
            points[:, :, 0] >= mask.shape[-1], points[:, :, 1] >= mask.shape[-2]
        )
        outside_img = torch.logical_or(outside_left, outside_right)
        points[outside_img] = -1

        outside_mask = mask[idx, points[:, :, 1], points[:, :, 0]] < 0.5
        outside = torch.logical_or(outside_img, outside_mask)

        points[outside] = -1
        point_masks[outside] = False
        setattr(self, name, points)
        setattr(self, f"{name}_mask", point_masks)

    def unproject(self, name, K, depth):
        points = getattr(self, name)
        assert points.shape[-1] == 2, "Should be 2d points"
        return unproject_points(points, K=K, depth=depth)

    def project(self, name, K):
        points = getattr(self, name)
        assert points.shape[-1] == 3, "Should be 3d points"
        return project_points(points, K=K)

    def apply_3D_transform(self, name, T):
        points3d = getattr(self, name)
        points3d = homogenuous(points3d)
        points3d = torch.matmul(T, points3d.permute(0, 2, 1)).permute(0, 2, 1)
        setattr(self, name, points3d[:, :, :3])

    def apply_affine(self, name, T):
        points2d = getattr(self, name)
        mask = points2d[:, :, 0] == -1
        points2d = homogenuous(points2d)
        points2d = torch.matmul(T, points2d.permute(0, 2, 1)).permute(0, 2, 1)
        points2d = points2d[:, :, :2] / points2d[:, :, 2:]
        points2d[mask] = -1
        setattr(self, name, points2d)


@dataclass
class KeyPointSampler:
    def __init__(self, tar_size=224, patch_size=14):
        self.tar_size = tar_size
        self.patch_size = patch_size
        self.numPatches_per_axis = int(tar_size / patch_size)
        self.init_points2d()

    def init_points2d(self):
        """
        Initialize the normalized center patch
        """
        x = torch.arange(0, self.tar_size, self.patch_size).float()
        x += self.patch_size / 2
        y = torch.arange(0, self.tar_size, self.patch_size).float()
        y += self.patch_size / 2

        yy, xx = torch.meshgrid(y, x)
        self.grid_points = torch.stack([yy.flatten(), xx.flatten()], dim=1)
        logger.info("Initialized normalized center patch done!")

    def convert_to_patch_coordinates(self, points):
        mask = points[:, :, 0] == -1
        patch_coords = points / self.patch_size
        patch_coords[mask] = -1
        return patch_coords

    def sample_pts(
        self,
        T_src2target,
        T_tar2source,
        src_data,
        tar_data,
    ):
        """
        1. Sample from cropped image, then convert to original image, then unproject to 3d
        2. Transform 3d points
        3. Project 3d points to 2d, then convert to cropped image
        4. Formatting the keypoints
        """
        assert isinstance(src_data, KeypointInput)
        assert isinstance(tar_data, KeypointInput)

        batch_size = T_src2target.shape[0]
        init_points = self.grid_points.clone().unsqueeze(0)
        init_points = init_points.repeat(batch_size, 1, 1)

        # sample 2d points
        key_pts2d = Keypoint(src=init_points.clone(), tar=init_points.clone())
        key_pts2d.mask("src", src_data.mask)
        key_pts2d.mask("tar", tar_data.mask)
        key_pts2d_cropped = key_pts2d.clone()

        # from crop img to original img: apply inverse(M)
        key_pts2d.apply_affine("src", inverse_affine(src_data.M))
        key_pts2d.apply_affine("tar", inverse_affine(tar_data.M))

        # unproject 2d points
        key_pts3d_src = key_pts2d.unproject(
            "src", K=src_data.K, depth=src_data.full_depth
        )
        key_pts3d_tar = key_pts2d.unproject(
            "tar", K=tar_data.K, depth=tar_data.full_depth
        )
        key_pts3d = Keypoint(src=key_pts3d_src, tar=key_pts3d_tar)

        # transform 3d points
        key_pts3d.apply_3D_transform("src", T_src2target)
        key_pts3d.apply_3D_transform("tar", T_tar2source)

        # reprojection
        reproj_key_pts2d_src = key_pts3d.project("src", K=tar_data.K)
        reproj_key_pts2d_tar = key_pts3d.project("tar", K=src_data.K)
        reproj_key_pts2d = Keypoint(src=reproj_key_pts2d_src, tar=reproj_key_pts2d_tar)

        # from original to crop
        reproj_key_pts2d.apply_affine("src", tar_data.M)
        reproj_key_pts2d.apply_affine("tar", src_data.M)

        # mask the source, target keypoints
        reproj_key_pts2d.mask("src", tar_data.mask)
        reproj_key_pts2d.mask("tar", src_data.mask)

        # invalid points are points outside the original and reprojected image
        mask_tar = key_pts2d_cropped.tar[:, :, 0] == -1
        mask_tar_reproj = reproj_key_pts2d.tar[:, :, 0] == -1
        mask_tar_all = torch.logical_or(mask_tar, mask_tar_reproj)

        mask_src = key_pts2d_cropped.src[:, :, 0] == -1
        mask_src_reproj = reproj_key_pts2d.src[:, :, 0] == -1
        mask_src_all = torch.logical_or(mask_src, mask_src_reproj)

        # compute distance between reproj(tar_pts) and src_pts
        for idx in range(batch_size):
            distance_ = torch.cdist(
                reproj_key_pts2d.tar[idx].float(), key_pts2d.src[idx].float()
            )
            distance_[mask_tar_all[idx]] = MAX_VALUES
            distance_[:, mask_src_all[idx]] = MAX_VALUES
            distance, _ = torch.min(distance_, dim=1)
            mask_ = distance < 1000.0

            reproj_key_pts2d.tar[idx, ~mask_] = -1
            key_pts2d_cropped.tar[idx, ~mask_] = -1

        # convert to patch coordinates
        src_patchs = self.convert_to_patch_coordinates(reproj_key_pts2d.tar)
        tar_patchs = self.convert_to_patch_coordinates(key_pts2d_cropped.tar)
        return {
            "src_pts": src_patchs,
            "tar_pts": tar_patchs,
        }
