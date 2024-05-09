"""
Usage:
pip install pyrender trimesh opencv-python scikit-image tqdm distinctipy
python -m src.scripts.vis_bop_onboarding_dynamic 
"""

from PIL import Image
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from bop_toolkit_lib import inout
import cv2

# import sys.path
from src.megapose.lib3d.transform import Transform
from src.megapose.datasets.object_dataset import RigidObjectDataset, RigidObject
from src.megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from src.megapose.panda3d_renderer.types import Panda3dLightData
from src.megapose.datasets.scene_dataset import (
    CameraData,
    ObjectData,
)
from src.megapose.utils.conversion import convert_scene_observation_to_panda3d
from src.libVis.pil import draw_contour
import os
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation

colors = distinctipy.get_colors(50)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    dataset_dir = (
        "/gpfsscratch/rech/tvi/uyb58rn/datasets/gigaPose_datasets/datasets/hope"
    )
    split = "onboarding_dynamic"

    dataset_dir = Path(dataset_dir)
    split_dir = dataset_dir / split
    scenes = sorted([x for x in split_dir.iterdir() if x.is_dir()])
    print(f"Found {scenes} scenes")

    # load CAD models
    model_infos = inout.load_json(dataset_dir / "models/models_info.json")
    objects = []
    for obj_id in tqdm(model_infos):
        obj_label = f"obj_{int(obj_id):06d}"

        cad_path = (dataset_dir / "models" / obj_label).with_suffix(".ply").as_posix()
        object = RigidObject(
            label=obj_id,
            mesh_path=cad_path,
            mesh_units="mm",
            scaling_factor=1,
        )
        objects.append(object)

    panda_renderer = Panda3dSceneRenderer(RigidObjectDataset(objects), verbose=False)

    # define light
    light_datas = [
        Panda3dLightData(
            light_type="ambient",
            color=((1.0, 1.0, 1.0, 1)),
        ),
    ]

    for scene_dir in tqdm(scenes):
        scene_id = int(scene_dir.stem[4:])
        scene_gt = inout.load_json(scene_dir / "scene_gt.json")
        scene_camera = inout.load_json(scene_dir / "scene_camera.json")

        vis_dir = dataset_dir / f"vis_{split}"
        vis_dir = Path("./tmp/")
        vis_dir.mkdir(exist_ok=True, parents=True)

        # render the gt of first image
        first_im_id = [k for k in scene_gt.keys()][0]
        img_path = scene_dir / f"rgb/{int(first_im_id):06d}.jpg"
        first_image = Image.open(img_path)
        first_rgb = np.array(first_image)

        # convert to gray image
        first_image = cv2.cvtColor(np.copy(first_image), cv2.COLOR_RGB2GRAY)
        first_image = cv2.cvtColor(first_image, cv2.COLOR_GRAY2RGB)

        im_K = np.asarray(scene_camera[first_im_id]["cam_K"]).reshape(3, 3)
        camera_data = CameraData(
            K=im_K,
            TWC=Transform(
                np.eye(3),
                np.zeros(3),
            ),
            resolution=first_image.shape[:2],
        )

        object_datas = []
        for obj_gt in scene_gt[first_im_id]:
            object_data = ObjectData(
                label=str(obj_gt["obj_id"]),
                TWO=Transform(
                    np.array(obj_gt["cam_R_m2c"]).reshape(3, 3),
                    np.array(obj_gt["cam_t_m2c"]) * 0.001,
                ),
                unique_id=obj_gt["obj_id"],
            )
            object_datas.append(object_data)

        camera_data_, obj_data_ = convert_scene_observation_to_panda3d(
            camera_data, object_datas
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

        first_image[renderings.binary_mask > 0, :] = renderings.rgb[
            renderings.binary_mask > 0, :
        ]
        first_image = draw_contour(
            first_image, renderings.binary_mask, color=(0, 255, 0), to_pil=False
        )

        # subsample only 20 images per scene
        im_ids = sorted([int(k) for k in scene_camera.keys()])
        im_ids = im_ids[:: int(len(im_ids) / 20)]
        print(im_ids)
        for im_id in tqdm(im_ids):
            img_path = scene_dir / f"rgb/{int(im_id):06d}.jpg"
            image = Image.open(img_path)
            rgb = np.array(image)
            gray_rgb = cv2.cvtColor(np.copy(rgb), cv2.COLOR_RGB2GRAY)
            gray_rgb = cv2.cvtColor(gray_rgb, cv2.COLOR_GRAY2RGB)

            mask_path = scene_dir / f"mask_visib/{int(im_id):06d}_{0:06d}.png"
            mask = Image.open(mask_path)
            mask = np.array(mask)

            gray_rgb[mask > 0, :] = rgb[mask > 0, :]
            gray_rgb = draw_contour(gray_rgb, mask, color=(255, 0, 0), to_pil=False)

            # concatenate original visualization from BOP
            vis = np.concatenate([first_image, gray_rgb], axis=1)
            vis = Image.fromarray(vis)
            save_path = vis_dir / f"{scene_dir.stem}_{int(im_id):06d}.jpg"
            vis.save(save_path)
