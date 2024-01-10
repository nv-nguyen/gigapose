import numpy as np


def add(pred, gt, pcd, scale):
    """
    Given a predicted pose and a ground truth pose, compute the ADD metric.
    pcd: Nx3
    pred: 4x4
    gt: 4x4
    """
    pred_pcd = np.matmul(pred[:3, :3], pcd.T) + pred[:3, 3].reshape(-1, 1) * scale
    gt_pcd = np.matmul(gt[:3, :3], pcd.T) + gt[:3, 3].reshape(-1, 1) * scale
    dist = np.linalg.norm(pred_pcd - gt_pcd, axis=0)
    return dist
