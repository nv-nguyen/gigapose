import numpy as np


def force_binary_mask(mask, threshold=0.0):
    mask = np.where(mask > threshold, 1, 0)
    return mask


def mask_to_rle(binary_mask):
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle.get("counts")

    last_elem = 0
    running_length = 0

    for i, elem in enumerate(binary_mask.ravel(order="F")):
        if elem == last_elem:
            pass
        else:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        running_length += 1

    counts.append(running_length)

    return rle


def compute_ious(pred_mask, gt_mask):
    # make sure the masks are binary
    pred_mask = np.array(pred_mask > 0.5, dtype=np.float32)
    gt_mask = np.array(gt_mask > 0.5, dtype=np.float32)
    assert np.max(pred_mask) <= 1.0 and np.min(pred_mask) >= 0.0
    assert np.max(gt_mask) <= 1.0 and np.min(gt_mask) >= 0.0, f"{gt_mask.shape}"
    intersections = np.sum(
        (np.expand_dims(pred_mask, 1) * np.expand_dims(gt_mask, 0)) > 0, axis=(2, 3)
    )
    unions = np.sum(
        (np.expand_dims(pred_mask, 1) + np.expand_dims(gt_mask, 0)) > 0, axis=(2, 3)
    )
    ious = intersections / unions
    assert np.all(ious >= 0.0) and np.all(
        ious <= 1.0
    ), f"{np.sum(intersections)}, {np.sum(unions)}"
    return ious
