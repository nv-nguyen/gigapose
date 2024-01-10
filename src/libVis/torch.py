import os
import cv2
import matplotlib
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import transforms
import torch

from torchvision.utils import save_image
from src.libVis.numpy import (
    create_edge_from_mask,
    plot_keypoints,
)

os.environ["MPLCONFIGDIR"] = os.getcwd() + "./tmp/"
np.random.seed(2022)
COLORS_SPACE = np.random.randint(0, 255, size=(1000, 3))
inv_rgb_transform = T.Compose(
    [
        T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        ),
    ]
)


def convert_cmap(tensor):
    b, h, w = tensor.shape
    ndarr = tensor.to("cpu", torch.uint8).numpy()
    output = torch.zeros((b, 3, h, w), device=tensor.device)
    for i in range(b):
        cmap = matplotlib.cm.get_cmap("magma")
        tmp = cmap(ndarr[i])[..., :3]
        data = transforms.ToTensor()(np.array(tmp)).to(tensor.device)
        output[i] = data
    return output


def save_tensor_to_image(tensor, path, image_size=None, nrow=4):
    if image_size is not None:
        tensor = F.interpolate(tensor, image_size, mode="bilinear", align_corners=False)
    save_image(tensor, path, nrow=nrow)


def add_border_to_tensor(tensor, border_color, border_size=5):
    if border_color == "green":
        border_color = torch.tensor([0, 1, 0], dtype=tensor.dtype, device=tensor.device)
    elif border_color == "red":
        border_color = torch.tensor([1, 0, 0], dtype=tensor.dtype, device=tensor.device)
    else:
        raise NotImplementedError
    # Add the top border
    size = tensor[:, :, :border_size, :].shape
    tensor[:, :, :border_size, :] = border_color.view(1, 3, 1, 1).expand(size)
    # Add the bottom border
    size = tensor[:, :, -border_size:, :].shape
    tensor[:, :, -border_size:, :] = border_color.view(1, 3, 1, 1).expand(size)
    # Add the left border
    size = tensor[:, :, :, :border_size].shape
    tensor[:, :, :, :border_size] = border_color.view(1, 3, 1, 1).expand(size)
    # Add the right border
    size = tensor[:, :, :, -border_size:].shape
    tensor[:, :, :, -border_size:] = border_color.view(1, 3, 1, 1).expand(size)
    return tensor


def put_image_to_grid(list_imgs, adding_margin=True):
    num_col = len(list_imgs)
    b, c, h, w = list_imgs[0].shape
    device = list_imgs[0].device
    if adding_margin:
        num_all_col = num_col + 1
    else:
        num_all_col = num_col
    grid = torch.zeros((b * num_all_col, 3, h, w), device=device).to(list_imgs[0].dtype)
    idx_grid = torch.arange(0, grid.shape[0], num_all_col, device=device).to(
        torch.int64
    )
    for i in range(num_col):
        grid[idx_grid + i] = list_imgs[i].to(list_imgs[0].dtype)
    return grid, num_col + 1


def resize_tensor(tensor, size):
    return F.interpolate(tensor, size, mode="bilinear", align_corners=True)


def convert_tensor_to_image(tensor, type="rgb", unnormalize=True):
    if unnormalize and type == "rgb":
        tensor = inv_rgb_transform(tensor)
    if type == "rgb":
        tmp = tensor.permute(0, 2, 3, 1) * 255
    elif type == "mask":
        tmp = tensor * 255
    return np.uint8(tmp.cpu().numpy())


def merge_image(images):
    if len(images) == 1:
        return images[0]
    else:
        np.concatenate(images, axis=0)


def plot_keypoints_batch(
    data,
    type_data="gt",
    unnormalize=True,
    patch_size=14,
    num_samples=16,
    concate_input_in_pred=True,
    write_num_matches=True,
):
    batch_size = data.src_img.shape[0]
    batch_size = min(batch_size, num_samples)

    # convert tensor to numpy
    src_img = convert_tensor_to_image(data.src_img, unnormalize=unnormalize)
    tar_img = convert_tensor_to_image(data.tar_img, unnormalize=unnormalize)

    if type_data == "gt":
        src_pts = data.src_pts.cpu().numpy()
        tar_pts = data.tar_pts.cpu().numpy()
    elif type_data == "pred":
        src_pts = data.pred_src_pts.cpu().numpy()
        tar_pts = data.pred_tar_pts.cpu().numpy()

    matching_imgs = []
    for idx in range(batch_size):
        mask = src_pts[idx, :, 0] != -1
        # mask = np.logical_and(mask, tar_pts[idx, :, 0] != -1)
        border_color = None  # [255, 0, 0]
        concate_input = concate_input_in_pred
        keypoint_img = plot_keypoints(
            src_img=src_img[idx],
            src_pts=src_pts[idx][mask],
            tar_img=tar_img[idx],
            tar_pts=tar_pts[idx][mask],
            border_color=border_color,
            concate_input=concate_input,
            write_num_matches=write_num_matches,
            patch_size=patch_size,
        )
        matching_imgs.append(torch.from_numpy(keypoint_img / 255.0).permute(2, 0, 1))
    matching_imgs = torch.stack(matching_imgs)
    return matching_imgs


def plot_Kabsch(batch, affine_transforms, unnormalize=True, num_img_plots=16):
    """
    Plot the mapping includes: rotation, translation, scale
    """
    batch_size = batch.src_img.shape[0]
    batch_size = min(batch_size, num_img_plots)
    affined_imgs = []

    # convert tensor to numpy
    src_imgs = convert_tensor_to_image(batch.src_img, unnormalize=unnormalize)
    src_masks = convert_tensor_to_image(batch.src_mask, type="mask")
    tar_imgs = convert_tensor_to_image(batch.tar_img, unnormalize=unnormalize)
    tar_masks = convert_tensor_to_image(batch.tar_mask, type="mask")
    Ms = affine_transforms.clone().cpu().numpy()

    for idx in range(batch_size):
        src_img, tar_img = src_imgs[idx], tar_imgs[idx]
        src_mask, tar_mask = src_masks[idx], tar_masks[idx]
        M = Ms[idx]
        src_rgba = np.concatenate([src_img, src_mask[:, :, np.newaxis]], axis=2)
        tar_mask = Image.fromarray(np.uint8(tar_mask))

        # convert tar_img to gray scale
        rows, cols, _ = tar_img.shape
        tar_img = cv2.cvtColor(tar_img, cv2.COLOR_RGB2GRAY)
        tar_img = cv2.cvtColor(tar_img, cv2.COLOR_GRAY2RGB)
        tar_img = Image.fromarray(np.uint8(tar_img))

        M = M[:2, :]
        wrap_src_rgba = cv2.warpAffine(np.uint8(src_rgba), M, (cols, rows))
        wrap_src_PIL = Image.fromarray(wrap_src_rgba)
        wrap_src_mask = wrap_src_PIL.getchannel("A")
        tar_img.paste(wrap_src_PIL.convert("RGB"), (0, 0), wrap_src_mask)

        wrap_src_egde = create_edge_from_mask(
            image_size=tar_img.size, mask=wrap_src_mask
        )
        query_edge = create_edge_from_mask(image_size=tar_img.size, mask=tar_mask)

        tar_img = np.array(tar_img.convert("RGB"))
        tar_img[wrap_src_egde, :] = [255, 0, 0]
        tar_img[query_edge, :] = [0, 255, 0]
        # overlay
        # query = cv2.addWeighted(
        #     np.array(query), 0.5, np.array(reproj_ref_PIL.convert("RGB")), 0.5, 0
        # )
        affined_imgs.append(torch.from_numpy(tar_img / 255.0).permute(2, 0, 1))
    affined_imgs = torch.stack(affined_imgs)
    return affined_imgs
