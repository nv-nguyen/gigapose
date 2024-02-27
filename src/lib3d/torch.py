import torch
from scipy.spatial.transform import Rotation
from einops import repeat
import torch.nn.functional as F


def affine_torch(rotation, scale=None, translation=None):
    if len(rotation.shape) == 2:
        """
        Create 2D affine transformation matrix
        """
        M = torch.eye(3, device=scale.device, dtype=scale.dtype)
        M[:2, :2] = rotation
        if scale is not None:
            M[:2, :2] *= scale
        if translation is not None:
            M[:2, 2] = translation
        return M
    else:
        Ms = torch.eye(3, device=scale.device, dtype=scale.dtype)
        Ms = Ms.unsqueeze(0).repeat(rotation.shape[0], 1, 1)
        Ms[:, :2, :2] = rotation
        if scale is not None:
            Ms[:, :2, :2] *= scale.unsqueeze(1).unsqueeze(1)
        if translation is not None:
            Ms[:, :2, 2] = translation
        return Ms


def homogenuous(pixel_points):
    """
    Convert pixel coordinates to homogenuous coordinates
    """
    device = pixel_points.device
    if len(pixel_points.shape) == 2:
        one_vector = torch.ones(pixel_points.shape[0], 1).to(device)
        return torch.cat([pixel_points, one_vector], dim=1)
    elif len(pixel_points.shape) == 3:
        one_vector = torch.ones(pixel_points.shape[0], pixel_points.shape[1], 1).to(
            device
        )
        return torch.cat([pixel_points, one_vector], dim=2)
    else:
        raise NotImplementedError


def inverse_affine(M):
    """
    Inverse 2D affine transformation matrix of cropping
    """
    if len(M.shape) == 2:
        M = M.unsqueeze(0)
    if len(M.shape) == 3:
        assert (M[:, 1, 0] == 0).all() and (M[:, 0, 1] == 0).all()
        assert (M[:, 0, 0] == M[:, 1, 1]).all(), f"M: {M}"

        scale = M[:, 0, 0]
        M_inv = torch.eye(3, device=M.device, dtype=M.dtype)
        M_inv = M_inv.unsqueeze(0).repeat(M.shape[0], 1, 1)
        M_inv[:, 0, 0] = 1 / scale  # scale
        M_inv[:, 1, 1] = 1 / scale  # scale
        M_inv[:, :2, 2] = -M[:, :2, 2] / scale.unsqueeze(1)  # translation
    else:
        raise ValueError("M must be 2D or 3D")
    return M_inv


def apply_affine(M, points):
    """
    M: (N, 3, 3)
    points: (N, 2)
    """
    if len(points.shape) == 2:
        transformed_points = torch.einsum(
            "bhc,bc->bh",
            M,
            homogenuous(points),
        )  # (N, 3)
        transformed_points = transformed_points[:, :2] / transformed_points[:, 2:]
    elif len(points.shape) == 3:
        transformed_points = torch.einsum(
            "bhc,bnc->bnh",
            M,
            homogenuous(points),
        )
        transformed_points = transformed_points[:, :, :2] / transformed_points[:, :, 2:]
    else:
        raise NotImplementedError
    return transformed_points


def unproject_points(points2d, K, depth):
    """
    Unproject points from 2D to 3D
    """

    idx = torch.arange(points2d.shape[0])[:, None].repeat(1, points2d.shape[1])
    points2d[:, :, 1] = torch.clamp(points2d[:, :, 1], 0, depth.shape[1] - 1)
    points2d[:, :, 0] = torch.clamp(points2d[:, :, 0], 0, depth.shape[2] - 1)
    depth1d = depth[idx, points2d[:, :, 1].long(), points2d[:, :, 0].long()]
    points3d = homogenuous(points2d).float()
    K_inv = torch.inverse(K).float()
    points3d = torch.matmul(K_inv, points3d.permute(0, 2, 1)).permute(0, 2, 1)
    points3d = points3d * depth1d.unsqueeze(-1)
    return points3d


def project_points(points3d, K):
    """
    Project points from 3D to 2D
    points_3d: (N, 3)
    """
    points2d = torch.matmul(K, points3d.permute(0, 2, 1)).permute(0, 2, 1)
    points2d = points2d[:, :, :2] / points2d[:, :, 2:]
    return points2d


def cosSin_inv(cos_sin_inplane, normalize=False):
    cos = cos_sin_inplane[:, 0]
    sin = cos_sin_inplane[:, 1]
    if normalize:
        cos = cos / (cos**2 + sin**2)
        sin = sin / (cos**2 + sin**2)
    angle = torch.atan2(sin, cos)
    return angle % (2 * torch.pi)


def cosSin(angle):
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)


def get_relative_scale_inplane(src_K, tar_K, src_pose, tar_pose, src_M, tar_M):
    """
    scale(source->target) = (ref_z / query_z) * (scale(query_M) / scale(ref_M)) * (ref_f / query_f)
    """
    relZ = src_pose[:, 2, 3] / tar_pose[:, 2, 3]
    relCrop = torch.norm(tar_M[:, :2, 0], dim=1) / torch.norm(src_M[:, :2, 0], dim=1)
    rel_focal = src_K[:, 0, 0] / tar_K[:, 0, 0]
    relScale = relZ * relCrop / rel_focal

    relR = torch.matmul(tar_pose[:, :3, :3], src_pose[:, :3, :3].transpose(1, 2))
    if relR.device == torch.device("cpu"):
        relativeR = Rotation.from_matrix(relR.numpy()).as_euler("zxy")
    else:
        relativeR = Rotation.from_matrix(relR.cpu().numpy()).as_euler("zxy")
    relInplane = torch.from_numpy(relativeR[:, 0]).float()
    return relScale, (relInplane + 2 * torch.pi) % (2 * torch.pi)


def normalize_affine_transform(transforms):
    """
    Input: Affine transformation
    Output: Normalized affine transformation
    """
    norm_transforms = torch.zeros_like(transforms)
    norm_transforms[:, :, 2, 2] = 1

    scale = torch.norm(transforms[:, :, :2, 0], dim=2)
    scale = repeat(scale, "b n -> b n h w", h=2, w=2)

    norm_transforms[:, :, :2, :2] = transforms[:, :, :2, :2] / scale
    return norm_transforms


def geodesic_distance(pred_cosSin, gt_cosSin, normalize=False):
    if normalize:
        pred_cosSin = F.normalize(pred_cosSin, dim=1)
        gt_cosSin = F.normalize(gt_cosSin, dim=1)
    pred_cos = pred_cosSin[:, 0]
    pred_sin = pred_cosSin[:, 1]
    gt_cos = gt_cosSin[:, 0]
    gt_sin = gt_cosSin[:, 1]
    cos_diff = pred_cos * gt_cos + pred_sin * gt_sin
    cos_diff = torch.clamp(cos_diff, -1, 1)
    loss = torch.acos(cos_diff).mean()
    return loss
