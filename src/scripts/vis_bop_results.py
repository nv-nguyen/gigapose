import nvisii as visii
from pathlib import Path
from bop_toolkit_lib import inout
import pyrender
import os
import logging
import pyrr
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm
import trimesh
import xatlas
from PIL import Image
import cv2
import numpy as np
from src.megapose.datasets.scene_dataset import (
    ObservationInfos,
    CameraData,
    ObjectData,
    SceneObservation,
)
from src.megapose.lib3d.transform import Transform
from src.megapose.datasets.object_dataset import RigidObjectDataset, RigidObject
from src.megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from src.megapose.panda3d_renderer.types import Panda3dLightData
from src.megapose.utils.conversion import convert_scene_observation_to_panda3d
from src.utils.trimesh import create_mesh_from_points, load_mesh
from scipy import spatial
import time
from src.libVis.numpy import get_cmap
from skimage.feature import canny
from skimage.morphology import binary_dilation

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
results = {
    "gpose": {
        "itodd": "gpose2023_itodd-test_2487a8e7-deaf-445c-a732-7229902c4745.csv",
        "hb": "gpose2023_hb-test_6da106fb-4a62-45b0-ab14-0118c75bc497.csv",
        "icbin": "gpose2023_icbin-test_a043bf3e-311a-40ac-916c-a893a81dd0e9.csv",
        "lmo": "gpose2023_lmo-test_87ee2d1b-1af9-4f6e-a51b-64b9a606222e.csv",
        "tless": "gpose2023_tless-test_f6f57fde-c17f-49a8-9859-3dad44d18212.csv",
        "ycbv": "gpose2023_ycbv-test_f1a3e861-761d-41ea-b585-e84f9fd40591.csv",
        "tudl": "gpose2023_tudl-test_e60be2c3-35a9-4a30-b563-83284be6672b.csv",
    },
    "genflow": {
        "itodd": "genflow-multihypo16_itodd-test_91722167-50d9-48ce-98cc-dc34d148d78c.csv",
        "hb": "genflow-multihypo16_hb-test_947cd992-6058-480b-80e6-01f09a4f431f.csv",
        "icbin": "genflow-multihypo16_icbin-test_5bcbb2fb-24a3-43df-982b-3fadc69bb727.csv",
        "lmo": "genflow-multihypo16_lmo-test_bb86b4a3-17ca-4a7a-b5ef-6d9007c62224.csv",
        "tless": "genflow-multihypo16_tless-test_1763da63-c69a-4926-a494-49b163c93ed3.csv",
        "ycbv": "genflow-multihypo16_ycbv-test_e77b7247-29c7-46bb-b468-11da990ca422.csv",
        "tudl": "genflow-multihypo16_tudl-test_833fb644-15bb-45fb-be38-4038fca1a36b.csv",
    },
}


def draw_contour(img_PIL, mask, color, to_pil=True):
    edge = canny(mask)
    edge = binary_dilation(edge, np.ones((2, 2)))
    img = np.array(img_PIL)
    img[edge, :] = color
    if to_pil:
        return Image.fromarray(img)
    else:
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


def group_by_image_level(data, image_key="im_id"):
    # group the detections by scene_id and im_id
    data_per_image = {}
    for det in data:
        scene_id, im_id = int(det["scene_id"]), int(det[image_key])
        key = f"{scene_id:06d}_{im_id:06d}"
        if key not in data_per_image:
            data_per_image[key] = []
        data_per_image[key].append(det)
    return data_per_image


def create_mesh_from_points(vertices, triangles, save_path):
    vmapping, indices, uvs = xatlas.parametrize(vertices, triangles)
    xatlas.export(save_path, vertices[vmapping], indices, uvs)
    return trimesh.load(save_path)


def set_up_camera(name="camera"):
    camera = visii.entity.create(
        name=name,
        transform=visii.transform.create(name),
        camera=visii.camera.create(
            name=name,
        ),
    )
    camera.get_transform().look_at(
        visii.vec3(0, 0, -1),  # look at (world coordinate)
        visii.vec3(0, 1, 0),  # up vector
        visii.vec3(0, 0, 0),  # camera_origin
    )
    visii.set_camera_entity(camera)
    visii.set_dome_light_intensity(1)
    visii.set_dome_light_color(visii.vec3(1, 1, 1))
    return camera


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


def add_obj(
    scale,
    name,
    cad_path="",
    texture_path=None,
):
    # check if name is already in the scene
    mesh = visii.mesh.create_from_file(name, cad_path)
    obj_entity = visii.entity.create(
        name=name,
        mesh=mesh,
        transform=visii.transform.create(name),
        material=visii.material.create(name),
    )
    # should randomize
    obj_entity.get_material().set_metallic(0)  # should 0 or 1
    obj_entity.get_material().set_transmission(0)  # should 0 or 1
    obj_entity.get_material().set_roughness(1)  # default is 1

    if texture_path is not None and os.path.exists(texture_path):
        obj_texture = visii.texture.create_from_file(name, texture_path)
        obj_entity.get_material().set_base_color_texture(obj_texture)

    obj_entity.get_transform().set_scale(visii.vec3(scale))
    obj_entity.get_transform().set_position((0, 0, 0))  # visii vec3)
    return obj_entity


def set_intrinsic(camera, K, img_size):
    fx, fy, cx, cy = (
        K[0][0],
        K[1][1],
        K[0][2],
        K[1][2],
    )
    cam = pyrender.IntrinsicsCamera(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )
    proj_matrix = cam.get_projection_matrix(img_size[0], img_size[1])
    proj_matrix = visii.mat4(
        proj_matrix.flatten()[0],
        proj_matrix.flatten()[1],
        proj_matrix.flatten()[2],
        proj_matrix.flatten()[3],
        proj_matrix.flatten()[4],
        proj_matrix.flatten()[5],
        proj_matrix.flatten()[6],
        proj_matrix.flatten()[7],
        proj_matrix.flatten()[8],
        proj_matrix.flatten()[9],
        proj_matrix.flatten()[10],
        proj_matrix.flatten()[11],
        proj_matrix.flatten()[12],
        proj_matrix.flatten()[13],
        proj_matrix.flatten()[14],
        proj_matrix.flatten()[15],
    )
    proj_matrix = visii.transpose(proj_matrix)
    camera.get_camera().set_projection(proj_matrix)
    logging.info("Set intrinsic done!")


def rotation2quaternion(rot):
    m = pyrr.Matrix33(
        [
            [rot[0], rot[1], rot[2]],
            [rot[3], rot[4], rot[5]],
            [rot[6], rot[7], rot[8]],
        ]
    )
    return m.quaternion


def set_object_pose(object_visii, object_pose, scale, color=None):
    if color is not None:
        object_visii.get_material().set_base_color(color)
    object_visii.get_material().set_roughness(1)
    object_visii.get_material().set_metallic(0)

    # set rotation
    quaternion = rotation2quaternion(object_pose[:3, :3].reshape(-1))
    object_visii.get_transform().set_rotation(
        visii.quat(
            quaternion[3],
            quaternion[0],
            quaternion[1],
            quaternion[2],
        )
    )
    # set translation
    position = object_pose[:3, 3]
    object_visii.get_transform().set_position(
        visii.vec3(
            position[0] * scale,
            position[1] * scale,
            position[2] * scale,
        )
    )

    # rotate coordinate system
    object_visii.get_transform().rotate_around(
        visii.vec3(0, 0, 0), visii.angleAxis(visii.pi(), visii.vec3(1, 0, 0))
    )


def filter_estimates(gt, pred_estimates):
    filtered_estimates = []
    for test_instance in gt:
        obj_id = test_instance["obj_id"]
        obj_estimate = [
            estimate for estimate in pred_estimates if estimate["obj_id"] == obj_id
        ]
        if len(obj_estimate) > 0:
            # keep the highest score
            obj_estimate = max(obj_estimate, key=lambda x: x["score"])
            filtered_estimates.append(obj_estimate)
    # sort by object_id
    # filtered_estimates = sorted(filtered_estimates, key=lambda x: x["obj_id"])
    return filtered_estimates


def set_texture_visii(
    meshes,
    obj_label,
    visii_mesh_pred,
    visii_mesh_gt,
    export_path,
    prefix,
    symmetry_obj_ids,
    max_distance=10,
    res_texture=1024,
):
    mesh = meshes[obj_label]
    distance = np.zeros(len(mesh.vertices))
    points_gt, points_pred = [], []
    for i, vertex in enumerate(mesh.vertices):
        v = visii.vec4(vertex[0], vertex[1], vertex[2], 1)
        point_gt = visii_mesh_gt.get_transform().get_local_to_world_matrix() * v
        point_pred = visii_mesh_pred.get_transform().get_local_to_world_matrix() * v
        points_gt.append([point_gt[0], point_gt[1], point_gt[2]])
        points_pred.append([point_pred[0], point_pred[1], point_pred[2]])
        distance[i] = visii.distance(point_gt, point_pred)

    distance_symmetry = spatial.distance_matrix(
        np.array(points_gt), np.array(points_pred), p=2
    ).min(axis=1)
    obj_id = int(obj_label)
    if obj_id in symmetry_obj_ids or len(symmetry_obj_ids) == 0:
        distance = np.append(distance_symmetry, [0, max_distance])
    else:
        distance = np.append(distance, [max_distance])
    distance /= max_distance
    colors = get_cmap(distance, "turbo")
    texture = get_texture_distance(
        colors,
        mesh,
        resolution=res_texture,
    )

    visual = trimesh.visual.TextureVisuals(
        uv=mesh.visual.uv,
        material=trimesh.visual.material.SimpleMaterial(image=texture),
        image=texture,
    )
    mesh.visual = visual
    mesh.export(export_path / "tmp.obj")
    visii_mesh_vis = visii.mesh.create_from_file(
        str(export_path / (str(obj_label) + "_vis" + prefix)),
        str(export_path / "tmp.obj"),
    )
    nvisii_tex = visii.texture.create_from_file(
        str(export_path / (str(obj_label) + "_vis_texture" + prefix)),
        str(export_path / "material_0.png"),
    )
    visii_mesh_gt.set_mesh(visii_mesh_vis)
    visii_mesh_gt.get_material().set_base_color_texture(nvisii_tex)

    return mesh, texture


def reset_visii(reset_names):
    # remove all textures existings and get all objects far away
    for names in reset_names:
        obj_visii = visii.entity.get(names)
        obj_visii.get_transform().set_position(
            visii.vec3(
                -10000,
                -10000,
                -10000,
            )
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results_dir = Path("/home/nguyen/Documents/datasets/bop23_report")
    run_visii = True

    for dataset_name in ["tless"]:  # "lmo", "tudl",
        if dataset_name == "ycbv":
            symmetry_obj_ids = [13, 18, 19, 20]
        elif dataset_name == "lmo":
            symmetry_obj_ids = [11, 12]
        else:
            symmetry_obj_ids = []
        save_dir = results_dir / "comparison" / dataset_name
        save_dir.mkdir(exist_ok=True, parents=True)

        # load target poses and group by image level
        target_path = results_dir / "target" / f"{dataset_name}.json"
        test_list = inout.load_json(target_path)
        test_list = group_by_image_level(test_list, image_key="im_id")

        # load estimates and group by image level
        estimates = {}
        for method in ["gpose", "genflow"]:
            path = results_dir / method / results[method][dataset_name]
            estimate = inout.load_bop_results(path)
            estimates[method] = group_by_image_level(estimate)

        with initialize(config_path="../../configs/"):
            cfg = compose(config_name="train.yaml")
        OmegaConf.set_struct(cfg, False)

        # initialize dataloader
        cfg.machine.batch_size = 1
        cfg.data.test.dataloader.batch_size = cfg.machine.batch_size
        cfg.data.test.dataloader.dataset_name = dataset_name
        test_dataset = instantiate(cfg.data.test.dataloader)
        split, _ = test_dataset.get_split_name(dataset_name)

        # initialize visii
        if run_visii:
            visii.initialize(headless=True)
            visii.enable_denoiser()
            visii_camera = set_up_camera(name="camera")

        # initialize meshes
        bop_dir = test_dataset.root_dir
        dataset_dir = bop_dir / test_dataset.dataset_name

        cad_name = (
            "models" if test_dataset.dataset_name != "tless" else "models_reconst"
        )
        cad_dir = dataset_dir / cad_name
        model_infos = inout.load_json(cad_dir / "models_info.json")
        model_infos = [{"obj_id": int(obj_id)} for obj_id in model_infos.keys()]

        cad_eval_dir = dataset_dir / "models_eval"
        objects = []
        for model_info in tqdm(model_infos):
            obj_id = int(model_info["obj_id"])
            obj_label = f"obj_{obj_id:06d}"

            cad_path = (cad_dir / obj_label).with_suffix(".ply").as_posix()
            object = RigidObject(
                label=str(model_info["obj_id"]),
                mesh_path=cad_path,
                mesh_units="mm",
                scaling_factor=1,
            )
            objects.append(object)

        panda_renderer = Panda3dSceneRenderer(
            RigidObjectDataset(objects), verbose=False
        )
        # define light
        light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=((1.0, 1.0, 1.0, 1)),
            ),
        ]

        if run_visii:
            meshes = {}
            mesh_vertices = {}
            cad_path = {}
            for model_info in tqdm(model_infos):
                obj_id = int(model_info["obj_id"])
                cad_eval_path = (
                    (cad_eval_dir / f"obj_{obj_id:06d}").with_suffix(".ply").as_posix()
                )
                cad_path[str(obj_id)] = cad_eval_path
                add_obj(
                    scale=0.1,
                    name=str(obj_id),
                    cad_path=cad_eval_path,
                    texture_path=None,
                )
                obj_visii = visii.entity.get(str(obj_id))

                # init far
                obj_visii.get_transform().set_position(
                    visii.vec3(
                        -10000,
                        -10000,
                        -10000,
                    )
                )

                new_cad_path = f"{save_dir}/{str(model_info['obj_id'])}.obj"
                mesh = create_mesh_from_points(
                    np.array(obj_visii.get_mesh().get_vertices()),
                    np.array(obj_visii.get_mesh().get_triangle_indices()).reshape(
                        -1, 3
                    ),
                    save_path=new_cad_path,
                )
                meshes[str(obj_id)] = mesh

                vertices = load_mesh(new_cad_path).apply_scale(0.1).vertices
                mesh_vertices[str(obj_id)] = np.array(vertices)

            logging.info(f"Setting up {len(model_infos)} objects for VISII !")

        # image key that available in both estimates and target
        avail_keys = (
            set(estimates["genflow"].keys())
            & set(estimates["gpose"].keys())
            & set(test_list.keys())
        )
        visii_obj_names = []
        for image_key in tqdm(avail_keys, desc="Rendering"):
            try:
                test_samples = test_list[image_key]
                # skip scene that have multiple instances of same object ID
                if len(set([obj["obj_id"] for obj in test_samples])) != len(
                    test_samples
                ):
                    continue

                scene_id, im_id = test_samples[0]["scene_id"], test_samples[0]["im_id"]
                img_path = dataset_dir / split / f"{scene_id:06d}/rgb/{im_id:06d}.png"
                img = Image.open(img_path)

                gt_info_path = (
                    dataset_dir / split / f"{scene_id:06d}/scene_gt_info.json"
                )
                gt_info = inout.load_json(gt_info_path)[f"{im_id}"]
                gt_cam_path = dataset_dir / split / f"{scene_id:06d}/scene_camera.json"
                gt_cam = inout.load_json(gt_cam_path)[f"{im_id}"]
                gt_path = dataset_dir / split / f"{scene_id:06d}/scene_gt.json"
                gt = inout.load_json(gt_path)[f"{im_id}"]
                # keep only gt in test list

                gt = [
                    gt[i]
                    for i in range(len(gt))
                    if gt[i]["obj_id"] in [sample["obj_id"] for sample in test_samples]
                ]
                # sort objects by translation to camera
                gt = sorted(gt, key=lambda x: -np.linalg.norm(x["cam_t_m2c"]))
                scene_infos = ObservationInfos(
                    scene_id=str(scene_id), view_id=str(im_id)
                )
                pred_object_datas = {}
                count = {}
                for method, estimate in estimates.items():
                    pred_object_datas[method] = []
                    pred_estimate = estimate[image_key]
                    pred_estimate = filter_estimates(gt, pred_estimate)
                    count[method] = len(pred_estimate) == len(gt)
                    for idx_obj in range(len(pred_estimate)):
                        object_data = ObjectData(
                            label=str(pred_estimate[idx_obj]["obj_id"]),
                            TWO=Transform(
                                np.array(pred_estimate[idx_obj]["R"]).reshape(3, 3),
                                np.array(pred_estimate[idx_obj]["t"]) * 0.001,
                            ),
                            unique_id=idx_obj,
                        )
                        pred_object_datas[method].append(object_data)
                if not (count["gpose"] and count["genflow"]):
                    continue

                object_datas = []
                for idx_obj in range(len(gt)):
                    object_data = ObjectData(
                        label=str(gt[idx_obj]["obj_id"]),
                        TWO=Transform(
                            np.array(gt[idx_obj]["cam_R_m2c"]).reshape(3, 3),
                            np.array(gt[idx_obj]["cam_t_m2c"]) * 0.001,
                        ),
                        unique_id=idx_obj,
                    )
                    object_datas.append(object_data)
                camera_data = CameraData(
                    K=np.array(gt_cam["cam_K"]).reshape(3, 3),
                    TWC=Transform(
                        np.eye(3),
                        np.zeros(3),
                    ),
                    resolution=np.array(img).shape[:2],
                )

                scene_obs = SceneObservation(
                    rgb=np.array(img),
                    object_datas=object_datas,
                    camera_data=camera_data,
                    infos=scene_infos,
                )
                # camera = visii.entity.get("camera")
                if run_visii:
                    set_intrinsic(
                        camera=visii_camera,
                        K=scene_obs.camera_data.K,
                        img_size=img.size,
                    )

                pred_reset_names = []
                gt_reset_names = []
                vis_imgs = [np.array(img)]
                save_path = save_dir / f"{image_key}"
                save_path.mkdir(exist_ok=True, parents=True)
                img.save(save_path / f"{image_key}_rgb.png")
                for method in ["gpose", "genflow"]:
                    if run_visii:
                        for idx_obj in range(len(pred_object_datas[method])):
                            obj_label = pred_object_datas[method][idx_obj].label
                            # set pred
                            if obj_label + f"_{method}" not in visii_obj_names:
                                add_obj(
                                    scale=0.1,
                                    name=obj_label + f"_{method}",
                                    cad_path=cad_path[obj_label],
                                    texture_path=None,
                                )
                                visii_obj_names.append(obj_label + f"_{method}")
                            if obj_label + "_gt" not in visii_obj_names:
                                add_obj(
                                    scale=0.1,
                                    name=obj_label + "_gt",
                                    cad_path=cad_path[obj_label],
                                    texture_path=None,
                                )
                                visii_obj_names.append(obj_label + "_gt")

                            obj_pose = pred_object_datas[method][
                                idx_obj
                            ].TWO.toHomogeneousMatrix()
                            obj_pose[:3, 3] *= 1000
                            obj_visii = visii.entity.get(obj_label + f"_{method}")
                            set_object_pose(
                                obj_visii,
                                obj_pose,
                                0.1,
                                color=visii.vec3(1, 0, 0),
                            )

                            gt_obj_pose = object_datas[
                                idx_obj
                            ].TWO.toHomogeneousMatrix()
                            gt_obj_pose[:3, 3] *= 1000
                            gt_obj_visii = visii.entity.get(obj_label + "_gt")
                            set_object_pose(
                                gt_obj_visii,
                                gt_obj_pose,
                                0.1,
                                color=visii.vec3(0, 1, 0),
                            )

                            set_texture_visii(
                                meshes=meshes,
                                obj_label=obj_label,
                                visii_mesh_pred=obj_visii,
                                visii_mesh_gt=gt_obj_visii,
                                symmetry_obj_ids=symmetry_obj_ids,
                                export_path=save_path,
                                prefix=f"_{method}",
                            )

                            pred_reset_names.append(obj_label + f"_{method}")
                            gt_reset_names.append(obj_label + "_gt")

                        reset_visii(reset_names=pred_reset_names)
                        visii.render_to_file(
                            width=img.size[0],
                            height=img.size[1],
                            samples_per_pixel=300,
                            file_path=f"{save_path}_heatmap_{method}.png",
                        )
                        img = Image.open(f"{save_path}_heatmap_{method}.png").convert(
                            "RGB"
                        )
                        vis_imgs.append(np.array(img))
                        reset_visii(reset_names=gt_reset_names)

                        # vis_imgs = Image.fromarray(vis_imgs)
                        # vis_imgs.save(save_dir / f"vis_{image_key}.png")
                        # print(save_path / f"{image_key}.png")

                # convert to panda3d format
                list_object_datas = [
                    object_datas,
                    pred_object_datas["gpose"],
                    pred_object_datas["genflow"],
                ]
                names = ["gt", "gpose", "genflow"]
                overlay_imgs = []
                for idx, obj_datas in enumerate(list_object_datas):
                    gray = cv2.cvtColor(
                        np.array(scene_obs.rgb.copy()), cv2.COLOR_RGB2GRAY
                    )
                    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                    for obj_data in obj_datas:
                        camera_data_, obj_data_ = convert_scene_observation_to_panda3d(
                            camera_data, [obj_data]
                        )

                        renderings = panda_renderer.render_scene(
                            object_datas=obj_data_,
                            camera_datas=[camera_data_],
                            light_datas=light_datas,
                            render_depth=True,
                            render_binary_mask=True,
                            render_normals=False,
                            copy_arrays=True,
                            clear=True,
                        )[0]

                        gray = mask_background(
                            gray_img=gray,
                            color_img=renderings.rgb,
                            masks=[renderings.binary_mask],
                            contour=True,
                            color=(0, 255, 0) if idx == 0 else (255, 0, 0),
                        )
                    gray_pil = Image.fromarray(gray)
                    gray_pil.save(save_path / f"{image_key}_overlay_{names[idx]}.png")
                    overlay_imgs.append(gray)
                overlay_imgs = np.concatenate(overlay_imgs, axis=1)

                if run_visii:
                    vis_imgs = np.concatenate(vis_imgs, axis=1)
                    all_imgs = np.concatenate([vis_imgs, overlay_imgs], axis=0)
                else:
                    all_imgs = overlay_imgs
                all_imgs = Image.fromarray(all_imgs)
                all_imgs.save(save_dir / f"{image_key}.png")
            except:
                pass
            print(save_dir / f"{image_key}.png")
        if run_visii:
            visii.deinitialize()
