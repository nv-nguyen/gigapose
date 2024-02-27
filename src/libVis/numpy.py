import os
from PIL import Image
import numpy as np
from skimage.feature import canny
from skimage.morphology import binary_dilation
import cv2
import time
import trimesh

os.environ["MPLCONFIGDIR"] = os.getcwd() + "./tmp/"
import matplotlib.pyplot as plt
from src.utils.logging import get_logger

logger = get_logger(__name__)
np.random.seed(2022)
COLORS_SPACE = np.random.randint(0, 255, size=(1000, 3))


def paint_texture(texture_image, min_val=240, max_val=255):
    """Paint the texture image to remove the white spots used in DiffDOPE"""
    # Convert the texture image to grayscale
    gray_texture = cv2.cvtColor(texture_image, cv2.COLOR_BGR2GRAY)
    # Find the white spots in the grayscale image (you may need to adjust the threshold)
    mask = cv2.inRange(gray_texture, min_val, max_val)
    # Apply inpainting to fill in the white spots
    texture_image = cv2.inpaint(
        texture_image, mask, inpaintRadius=1, flags=cv2.INPAINT_NS
    )
    return texture_image


def get_cmap(np_img, name_cmap):
    # cmap = matplotlib.colormaps.get_cmap(name_cmap)
    # tmp = cmap(np_img)
    # cmap = plt.cm.get_cmap('Reds')
    cmap = plt.cm.get_cmap("turbo")
    # Map the scalar values to colors using the interpolate function
    colors = trimesh.visual.color.interpolate(np_img, color_map=cmap)
    return colors


def add_border(image, color=[255, 0, 0], border_size=5):
    image[:border_size, :] = color
    image[-border_size:, :] = color

    # Add the border to the left and right columns
    image[:, :border_size] = color
    image[:, -border_size:] = color
    return image


def create_edge_from_mask(image_size, mask, size=3):
    tmp = Image.new("L", image_size, 0)
    tmp.paste(
        mask,
        (0, 0),
        mask,
    )
    edge = canny(np.array(tmp))
    edge = binary_dilation(edge, np.ones((size, size)))
    return edge


def write_text_on_image(image, text, color=[255, 0, 0]):
    image_size = image.shape[:2]
    # write text on top left corner
    position = (image_size[1] // 15, image_size[0] // 10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    cv2.putText(
        image,
        text,
        position,
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def plot_keypoints(
    src_img,
    src_pts,
    tar_img,
    tar_pts,
    border_color,
    patch_size=14,
    concate_input=True,
    write_num_matches=True,
):
    if patch_size != 1:
        src_pts = np.array(src_pts) * patch_size  # + patch_size * 0.5
        tar_pts = np.array(tar_pts) * patch_size  # + patch_size * 0.5
    src_pts = [cv2.KeyPoint(x, y, 1) for x, y in np.float32(src_pts)]
    tar_pts = [cv2.KeyPoint(x, y, 1) for x, y in np.float32(tar_pts)]

    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
    matched_img = cv2.drawMatchesKnn(
        img1=src_img,
        keypoints1=src_pts,
        img2=tar_img,
        keypoints2=tar_pts,
        matches1to2=[matches],
        outImg=None,
        flags=2,
    )
    if border_color is not None:
        matched_img = add_border(matched_img, color=border_color)
    if write_num_matches:
        write_text_on_image(image=matched_img, text=f"{len(matches)} matches")
    if concate_input:
        input_imgs = np.concatenate([src_img, tar_img], axis=1)
        matched_img = np.concatenate([input_imgs, matched_img], axis=0)
    return matched_img


def plot_pointCloud(cvImg, obj_pcd, matrix4x4, intrinsic, color=(255, 0, 0)):
    pts = np.matmul(
        intrinsic,
        np.matmul(matrix4x4[:3, :3], obj_pcd.T) + matrix4x4[:3, 3].reshape(-1, 1),
    )
    xs = pts[0] / (pts[2] + 1e-8)
    ys = pts[1] / (pts[2] + 1e-8)
    for pIdx in range(len(xs)):
        cvImg = cv2.circle(
            cvImg,
            (int(xs[pIdx]), int(ys[pIdx])),
            2,
            (int(color[0]), int(color[1]), int(color[2])),
            -1,
        )
    return cvImg


def compute_image(im, triangles, colors):
    # Specify (x,y) triangle vertices
    image_height = im.shape[0]
    a = triangles[0]
    b = triangles[1]
    c = triangles[2]

    # Specify colors
    red = colors[0]
    green = colors[1]
    blue = colors[2]

    # Make array of vertices
    # ax bx cx
    # ay by cy
    #  1  1  1
    triArr = np.asarray([a[0], b[0], c[0], a[1], b[1], c[1], 1, 1, 1]).reshape((3, 3))

    # Get bounding box of the triangle
    xleft = min(a[0], b[0], c[0])
    xright = max(a[0], b[0], c[0])
    ytop = min(a[1], b[1], c[1])
    ybottom = max(a[1], b[1], c[1])

    # Build np arrays of coordinates of the bounding box
    xs = range(xleft, xright)
    ys = range(ytop, ybottom)
    xv, yv = np.meshgrid(xs, ys)
    xv = xv.flatten()
    yv = yv.flatten()

    # Compute all least-squares /
    p = np.array([xv, yv, [1] * len(xv)])
    alphas, betas, gammas = np.linalg.lstsq(triArr, p, rcond=-1)[0]

    # Apply mask for pixels within the triangle only
    mask = (alphas >= 0) & (betas >= 0) & (gammas >= 0)
    alphas_m = alphas[mask]
    betas_m = betas[mask]
    gammas_m = gammas[mask]
    xv_m = xv[mask]
    yv_m = yv[mask]

    def mul(a, b):
        # Multiply two vectors into a matrix
        return np.asmatrix(b).T @ np.asmatrix(a)

    # Compute and assign colors
    colors = mul(red, alphas_m) + mul(green, betas_m) + mul(blue, gammas_m)
    try:
        im[(image_height - 1) - yv_m, xv_m] = colors
    except:
        pass
    return im


def compute_image_batch(resolution, triangles, colors):
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]

    # Specify colors
    red = colors[:, 0]
    green = colors[:, 1]
    blue = colors[:, 2]

    # Make array of vertices
    # ax bx cx
    # ay by cy
    #  1  1  1
    n = len(a[:, 0])
    one_vec = np.ones(n)
    triArr = np.asarray(
        [
            a[:, 0],
            b[:, 0],
            c[:, 0],
            a[:, 1],
            b[:, 1],
            c[:, 1],
            one_vec,
            one_vec,
            one_vec,
        ]
    ).reshape((n, 3, 3))

    # Get bounding box of the triangle
    xleft = np.minimum(np.minimum(a[:, 0], b[:, 0]), c[:, 0])
    xright = np.maximum(np.maximum(a[:, 0], b[:, 0]), c[:, 0])
    ytop = np.minimum(np.minimum(a[:, 1], b[:, 1]), c[:, 1])
    ybottom = np.maximum(np.maximum(a[:, 1], b[:, 1]), c[:, 1])

    # Build np arrays of coordinates of the bounding box
    x_ranges = [np.arange(left, right) for left, right in zip(xleft, xright)]
    y_ranges = [np.arange(top, bottom) for top, bottom in zip(ytop, ybottom)]
    xv = [np.meshgrid(x, y)[0] for x, y in zip(x_ranges, y_ranges)]
    xv = np.stack(xv).reshape((n, -1))
    yv = [np.meshgrid(x, y)[1] for x, y in zip(x_ranges, y_ranges)]
    yv = np.stack(yv).reshape((n, -1))

    # Compute all least-squares /
    p = np.stack([xv, yv, np.ones_like(xv)])
    alphas, betas, gammas = np.linalg.lstsq(triArr, p, rcond=-1)[0]

    # Apply mask for pixels within the triangle only
    mask = (alphas >= 0) & (betas >= 0) & (gammas >= 0)
    alphas_m = alphas[mask]
    betas_m = betas[mask]
    gammas_m = gammas[mask]
    xv_m = xv[mask]
    yv_m = yv[mask]
    # Compute and assign colors
    colors = red.T @ alphas_m + green.T @ betas_m + blue.T @ gammas_m

    texture = np.ones((resolution, resolution, 3), dtype=np.uint8) * 255
    texture[(resolution - 1) - yv_m, xv_m] = colors
    return texture


def get_texture_distance(colors, mesh, resolution):
    start_time = time.time()
    texture_image = (
        np.ones((resolution, resolution, 3), dtype=np.uint8) * 255
    )  # Initialize the texture image (resxres resolution)
    uvs = mesh.visual.uv
    for triangle in mesh.faces:
        # print(triangle)
        c1, c2, c3 = triangle
        uv1 = (
            int(uvs[c1][0] * resolution),
            int(uvs[c1][1] * resolution),
        )
        uv2 = (
            int(uvs[c2][0] * resolution),
            int(uvs[c2][1] * resolution),
        )
        uv3 = (
            int(uvs[c3][0] * resolution),
            int(uvs[c3][1] * resolution),
        )
        # fill in the colors
        c_filling = [
            (int(colors[c1][0]), int(colors[c1][1]), int(colors[c1][2])),
            (int(colors[c2][0]), int(colors[c2][1]), int(colors[c2][2])),
            (int(colors[c3][0]), int(colors[c3][1]), int(colors[c3][2])),
        ]
        texture_image = compute_image(texture_image, [uv1, uv2, uv3], c_filling)
    # logger.info("Time to compute texture distance: {}".format(time.time() - start_time))
    texture = paint_texture(texture_image)
    return Image.fromarray(texture)


def get_texture_distance_batch(colors, mesh, resolution):
    start_time = time.time()
    uvs = mesh.visual.uv

    # Scale and convert uvs based on resolution
    uv_scaled = (uvs[mesh.faces] * resolution).astype(int)
    uv_scaled = np.array(uv_scaled)

    # Convert colors and reshape
    c_filling = colors[mesh.faces].astype(int)
    c_filling = np.array(c_filling)

    texture = compute_image_batch(resolution, uv_scaled, c_filling)
    texture = paint_texture(texture)
    return Image.fromarray(texture)


if __name__ == "__main__":
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    write_text_on_image(image, "Hello")
    image_PIL = Image.fromarray(image)
    image_PIL.save("tmp.png")
