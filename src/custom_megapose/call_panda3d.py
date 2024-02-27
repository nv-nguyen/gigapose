from megapose.datasets.object_dataset import RigidObjectDataset, RigidObject
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.panda3d_renderer.types import Panda3dLightData
import logging
import numpy as np
from megapose.lib3d.transform import Transform
from PIL import Image
import os
import argparse
from src.utils.trimesh import get_obj_diameter
from bop_toolkit_lib import inout

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("cad_path", nargs="?", help="Path to the model file")
    parser.add_argument("obj_pose", nargs="?", help="Path to the model file")
    parser.add_argument(
        "output_dir", nargs="?", help="Path to where the final files will be saved"
    )
    parser.add_argument("gpus_devices", nargs="?", help="GPU devices")
    parser.add_argument("disable_output", nargs="?", help="Disable output of blender")
    parser.add_argument(
        "scale_translation", nargs="?", help="scale translation to meter"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus_devices)
    os.environ["EGL_VISIBLE_DEVICES"] = str(args.gpus_devices)

    label = 0
    is_shapeNet = "shapenet" in args.cad_path
    mesh_units = get_obj_diameter(args.cad_path) if not is_shapeNet else 1.0
    mesh_units = "m" if (mesh_units < 10 or is_shapeNet) else "mm"

    object = RigidObject(label=label, mesh_path=args.cad_path, mesh_units=mesh_units)
    rigid_object_dataset = RigidObjectDataset([object])

    # define camera
    disable_output = args.disable_output == "true" or args.disable_output == True
    renderer = Panda3dSceneRenderer(rigid_object_dataset, verbose=not disable_output)
    K = np.array([572.4114, 0.0, 320, 0.0, 573.57043, 240, 0.0, 0.0, 1.0]).reshape(
        (3, 3)
    )
    camera_pose = np.eye(4)
    # camera_pose[:3, 3] = -object_center
    TWC = Transform(camera_pose)
    camera_data = CameraData(K=K, TWC=TWC, resolution=(480, 640))

    # define light
    light_datas = [
        Panda3dLightData(
            light_type="ambient",
            color=((1.0, 1.0, 1.0, 1)),
        ),
    ]

    # load object poses
    object_poses = np.load(args.obj_pose)
    if mesh_units == "m" or args.scale_translation == "true":
        object_poses[:, :3, 3] /= 1000.0

    for idx_view in range(len(object_poses)):
        TWO = Transform(object_poses[idx_view])
        object_datas = [ObjectData(label=label, TWO=TWO)]
        camera_data, object_datas = convert_scene_observation_to_panda3d(
            camera_data, object_datas
        )

        # render
        renderings = renderer.render_scene(
            object_datas,
            [camera_data],
            light_datas,
            render_depth=True,
            render_binary_mask=True,
            render_normals=True,
            copy_arrays=True,
            clear=True,
        )[0]
        # save rgba
        rgb = renderings.rgb
        mask = renderings.binary_mask * 255
        rgba = np.concatenate([rgb, mask[:, :, None]], axis=2)
        save_path = f"{args.output_dir}/{idx_view:06d}.png"
        Image.fromarray(np.uint8(rgba)).save(save_path)

        # save depth
        save_depth_path = f"{args.output_dir}/{idx_view:06d}_depth.png"
        if mesh_units == "m" or args.scale_translation == "true":
            renderings.depth *= 1000.0
        inout.save_depth(save_depth_path, renderings.depth)

    # count the number of rendering
    renderings = os.listdir(args.output_dir)
    renderings = [r for r in renderings if r.endswith(".png")]
    num_rendering = len(renderings)
    # print(f"Found {num_rendering}", args.output_dir)
    if len(object_poses) * 2 != num_rendering:
        print("Warning: the number of rendering is not equal to the number of poses")
