import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import time
from functools import partial
import multiprocessing
import hydra
import glob
from pathlib import Path
from bop_toolkit_lib import inout

from src.lib3d.template_transform import get_obj_poses_from_template_level
from src.utils.inout import save_json
from src.utils.trimesh import get_obj_diameter, get_obj_center
from src.utils.logging import get_logger

logger = get_logger(__name__)


def call_get_obj_diameter(idx, cad_dir, model_infos):
    model_info = model_infos[idx]
    cad_id = model_info["gso_id"]
    cad_name = f"{cad_id}/meshes/model.obj"

    cad_path = osp.join(cad_dir, "models_normalized", cad_name)
    if not osp.exists(cad_path):
        diameter = -1
        object_center = np.ones(3) * -1
        print("Not found", cad_path)
    else:
        diameter = get_obj_diameter(cad_path) * 1000
        object_center = get_obj_center(cad_path) * 1000
        # print(diameter, object_center)
    model_info["diameter"] = diameter
    model_info["object_center"] = object_center.tolist()
    model_info["relative_cad_path"] = cad_name
    return model_info


def call(
    idx_obj,
    list_cad_path,
    list_output_dir,
    list_obj_pose_path,
    disable_output,
    num_gpus,
):
    output_dir = list_output_dir[idx_obj]
    cad_path = list_cad_path[idx_obj]
    obj_pose_path = list_obj_pose_path[idx_obj]
    if os.path.exists(
        output_dir
    ):  # remove first to avoid the overlapping folder of blender proc
        os.system("rm -r {}".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gpus_device = idx_obj % num_gpus
    os.makedirs(output_dir, exist_ok=True)
    cmd = f"python -m src.custom_megapose.call_panda3d {cad_path} {obj_pose_path} {output_dir} {gpus_device}"
    if disable_output:
        cmd += " true"
    else:
        cmd += " false"
    os.system(cmd)

    # make sure the number of rendered images is correct
    num_images = len(glob.glob(f"{output_dir}/*.png"))
    if num_images == len(np.load(obj_pose_path)) * 2:
        return True
    else:
        logger.info(f"Found {num_images}. Error {cad_path} {obj_pose_path}")
        return False


def re_pose_object(object_center, obj_poses):
    """
    Re-pose object to the origin given the rotation
    """
    new_object_poses = []
    for idx in range(len(obj_poses)):
        obj_pose = obj_poses[idx]
        obj_pose[:3, 3] -= obj_pose[:3, :3].dot(object_center)
        new_object_poses.append(obj_pose)
    return np.array(new_object_poses)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train",
)
def render(cfg) -> None:
    num_gpus = (
        len(cfg.machine.trainer.logger.devices)
        if "devices" in cfg.machine.trainer.logger
        else 1
    )
    num_workers = int(cfg.machine.num_workers)
    
    root_dir = Path(cfg.data.train.root_dir)
    root_save_dir = root_dir / "templates/gso"
    cad_dir = root_dir / "gso/models"
    model_infos = inout.load_json(root_dir / "gso/models_info.json")
    logger.info(f"Rendering {len(model_infos)} models!")

    os.makedirs(root_save_dir / "object_poses", exist_ok=True)
    template_poses = get_obj_poses_from_template_level(level=1, pose_distribution="all")

    logger.info(f"Using {num_gpus} GPUs")
    pool = multiprocessing.Pool(processes=int(cfg.machine.num_workers))
    call_with_index = partial(
        call_get_obj_diameter,
        cad_dir=cad_dir,
        model_infos=model_infos,
    )
    values = list(
        tqdm(
            pool.imap_unordered(call_with_index, range(len(model_infos))),
            total=len(model_infos),
        )
    )
    info_and_diameters = []
    for value in values:
        info_and_diameters.append(value)
    assert len(info_and_diameters) == len(
        model_infos
    ), f"Error in diameter {len(info_and_diameters)} {len(model_infos)}"

    save_json(
        path=root_save_dir / "diameter.json",
        info=info_and_diameters,
    )

    # render
    start_time = time.time()
    list_cad_paths, list_obj_pose_paths, list_output_dirs = [], [], []
    for model_info in tqdm(info_and_diameters):
        diameter = model_info["diameter"]
        object_center = model_info["object_center"]
        if diameter == -1:
            continue
        obj_id = model_info["obj_id"]
        cad_path = osp.join(
            cad_dir,
            "models_normalized",
            # "models_bop-renderer_scale=0.1",
            model_info["relative_cad_path"],
        )
        list_cad_paths.append(cad_path)
        obj_poses = template_poses.copy()
        obj_poses[:, :3, 3] *= diameter / 1000.0
        obj_poses = re_pose_object(object_center, obj_poses)
        obj_pose_path = f"{root_save_dir}/object_poses/{obj_id:06d}.npy"
        np.save(obj_pose_path, obj_poses)
        list_obj_pose_paths.append(obj_pose_path)

        output_dir = f"{root_save_dir}/{obj_id:06d}"
        list_output_dirs.append(output_dir)

    list_cad_paths, list_obj_pose_paths, list_output_dirs = (
        np.array(list_cad_paths),
        np.array(list_obj_pose_paths),
        np.array(list_output_dirs),
    )

    logger.info("Start rendering for {} objects".format(len(list_cad_paths)))
    start_time = time.time()
    pool = multiprocessing.Pool(processes=num_workers)
    call_ = partial(
        call,
        list_cad_path=list_cad_paths,
        list_output_dir=list_output_dirs,
        list_obj_pose_path=list_obj_pose_paths,
        disable_output=True,
        num_gpus=num_gpus,
    )
    values = list(
        tqdm(
            pool.imap_unordered(call_, range(len(list_cad_paths))),
            total=len(list_cad_paths),
        )
    )
    correct_values = [val for val in values]
    logger.info(f"Finished for {len(correct_values)}/{len(list_cad_paths)} objects")
    finish_time = time.time()
    logger.info(f"Total time {len(list_cad_paths)}: {finish_time - start_time}")


if __name__ == "__main__":
    render()
