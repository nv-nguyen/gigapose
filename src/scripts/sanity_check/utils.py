import logging
import os
from bop_toolkit_lib import inout
from pathlib import Path
import torchvision.transforms as T
from src.utils.logging import get_logger
from src.utils.gpu import assign_gpu
from src.custom_megapose.web_scene_dataset import (
    IterableWebSceneDataset,
    WebSceneDataset,
)
from tqdm import tqdm
from hydra.experimental import compose, initialize
from torch.utils.data import DataLoader

from src.megapose.datasets.object_dataset import RigidObjectDataset, RigidObject
from src.megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from src.megapose.panda3d_renderer.types import Panda3dLightData
from src.custom_megapose.transform import Transform
import trimesh
from src.utils.trimesh import load_mesh
from src.lib3d.numpy import matrix4x4
from src.libVis.numpy import plot_pointCloud

logger = get_logger(__name__)
assign_gpu()


def load_rigid_object_dataset(root_dir, cad_default_path):
    model_infos = inout.load_json(root_dir / "model_infos.json")

    # set up renderer and cads
    objects = []
    for model_info in tqdm(model_infos):
        cad_path = str(cad_default_path)
        cad_path = cad_path.replace("GSO_MODEL_ID", model_info["gso_id"])
        object = RigidObject(
            label=str(model_info["obj_id"]),
            mesh_path=cad_path,
            mesh_units="m",
            scaling_factor=0.1,
        )
        objects.append(object)
    rigid_object_dataset = RigidObjectDataset(objects)
    return rigid_object_dataset


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    from src.megapose.utils.conversion import convert_scene_observation_to_panda3d

    with initialize(config_path="../../configs/"):
        cfg = compose(config_name="train.yaml")
    root_dir = Path(cfg.data.train.root_dir)
    save_dir = "./tmp/sample"
    os.makedirs(save_dir, exist_ok=True)
    wds_dir = root_dir / "gso"
    web_dataset = WebSceneDataset(wds_dir, load_depth=True, load_segmentation=True)
    web_dataloader = IterableWebSceneDataset(web_scene_dataset=web_dataset)

    bop_cad_default_path = (
        root_dir
        / "megapose/models/google_scanned_objects/models_bop-renderer_scale=0.1/GSO_MODEL_ID/meshes/model.ply"
    )
    cad_default_path = (
        root_dir
        / "megapose/models/google_scanned_objects/models_normalized/GSO_MODEL_ID/meshes/model.obj"
    )
    bop_rigid_object_dataset = load_rigid_object_dataset(wds_dir, bop_cad_default_path)
    rigid_object_dataset = load_rigid_object_dataset(wds_dir, cad_default_path)
    renderer = Panda3dSceneRenderer(rigid_object_dataset, verbose=False)
    # define light
    light_datas = [
        Panda3dLightData(
            light_type="ambient",
            color=((1.0, 1.0, 1.0, 1)),
        ),
    ]
    colors = np.random.randint(0, 255, size=(1500, 3))

    for idx, scene_obs in enumerate(web_dataloader):
        rgb = scene_obs.rgb
        for object_data in scene_obs.object_datas:
            obj = bop_rigid_object_dataset.get_object_by_label(object_data.label)
            mesh = load_mesh(
                obj.mesh_path,
                ORIGIN_GEOMETRY=None,
            )
            # mesh.apply_scale(scale_factor)
            obj_keypoints = trimesh.sample.sample_surface(mesh, 1000)[0]
            matrix4x4 = object_data.TWO.toHomogeneousMatrix()
            # matrix4x4[:3, 3] /= 1000.0
            rgb = plot_pointCloud(
                rgb,
                obj_keypoints,
                matrix4x4,
                scene_obs.camera_data.K,
                color=colors[int(object_data.label)],
            )
        Image.fromarray(rgb).save(f"./tmp/{idx}.png")

        # scene_obs = scene_obs[0]
        scene_obs.camera_data.TWC = Transform(np.eye(4))

        for obj_data in scene_obs.object_datas:
            matrix4x4 = obj_data.TWO.toHomogeneousMatrix()
            matrix4x4[:3, 3] /= 1000
            obj_data.TWO = Transform(matrix4x4)

        visible_object_datas = []
        for object_data in scene_obs.object_datas:
            visible_object_datas.append(object_data)

        camera_data, object_datas = convert_scene_observation_to_panda3d(
            scene_obs.camera_data, visible_object_datas
        )

        renderings = renderer.render_scene(
            object_datas=object_datas,
            camera_datas=[camera_data],
            light_datas=light_datas,
            render_depth=True,
            render_binary_mask=False,
            render_normals=False,
            copy_arrays=True,
            clear=True,
        )[0]

        rgb = renderings.rgb
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(scene_obs.rgb)
        plt.subplot(1, 2, 2)
        plt.imshow(rgb)
        plt.savefig(f"./tmp/{idx}.png")

        if idx == 2:
            break
