from PIL import Image, ImageDraw
import cv2
import numpy as np
from skimage.feature import canny
from skimage.morphology import binary_dilation


def open_image(path, inplane=None):
    image = Image.open(path)
    if inplane is not None:
        image = image.rotate(inplane)
    return image


def draw_box(img_PIL, box, color="red", width=3):
    draw = ImageDraw.Draw(img_PIL)
    draw.rectangle(
        (
            (box[0], box[1]),
            (box[2], box[3]),
        ),
        outline=color,
        width=width,
    )
    return img_PIL


def draw_points(img_PIL, points_2d, color="blue"):
    draw = ImageDraw.Draw(img_PIL)
    for point in points_2d:
        draw.rectangle(
            ((point[0] - 0.1, point[1] + 0.1), (point[0] - 0.1, point[1] + 0.1)),
            outline=color,
            width=5,
        )
    return img_PIL


def draw_contour(img_PIL, mask, color, to_pil=True):
    edge = canny(mask)
    edge = binary_dilation(edge, np.ones((2, 2)))
    img = np.array(img_PIL)
    img[edge, :] = color
    if to_pil:
        return Image.fromarray(img)
    else:
        return img


def overlay_mask_on_rgb(rgb, mask, gray=False, color=(255, 0, 0), alpha=0.5):
    if gray:
        gray = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        img = np.array(rgb)
    r = int(color[0])
    g = int(color[1])
    b = int(color[2])
    img[mask, 0] = alpha * r + (1 - alpha) * img[mask, 0]
    img[mask, 1] = alpha * g + (1 - alpha) * img[mask, 1]
    img[mask, 2] = alpha * b + (1 - alpha) * img[mask, 2]
    edge = canny(mask)
    edge = binary_dilation(edge, np.ones((1, 1)))
    img[edge, :] = 255
    return img


def mask_background(gray_img, color_img, masks, color=(255, 0, 0), contour=True):
    """
    Put the color in the gray image according to the mask and add the contour
    """
    if isinstance(gray_img, Image.Image):
        gray_img = np.array(gray_img)
    for mask in masks:
        gray_img[mask > 0, :] = color_img[mask > 0, :]
        if contour:
            gray_img = draw_contour(gray_img, mask, color=color, to_pil=False)
    return gray_img
