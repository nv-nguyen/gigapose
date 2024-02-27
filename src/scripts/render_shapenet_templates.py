import os
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
from src.utils.logging import get_logger

logger = get_logger(__name__)


def call_panda3d(
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
    command = f"python -m src.custom_megapose.call_panda3d {cad_path} {obj_pose_path} {output_dir} {gpus_device}"
    if disable_output:
        command += " true"
    else:
        command += " false"
    os.system(command)

    # make sure the number of rendered images is correct
    num_images = len(glob.glob(f"{output_dir}/*.png"))
    if num_images == len(np.load(obj_pose_path)) * 2:
        return True
    else:
        logger.info(
            f"Found only {num_images}. Error in rendering {cad_path} {obj_pose_path}"
        )
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
    model_infos = inout.load_json(root_dir / "shapenet/models_info.json")
    logger.info(f"Rendering {len(model_infos)} models!")
    cad_dir = root_dir / "shapenet/models"

    root_save_dir = root_dir / "templates/shapenet"
    os.makedirs(root_save_dir, exist_ok=True)
    os.makedirs(root_save_dir / "object_poses", exist_ok=True)
    os.makedirs(root_save_dir / "object_poses_with_offset", exist_ok=True)
    template_poses = get_obj_poses_from_template_level(level=1, pose_distribution="all")

    if cfg.start_idx is None:
        cfg.start_idx = 0
    if cfg.end_idx is None:
        cfg.end_idx = len(model_infos)
        
    pool = multiprocessing.Pool(processes=int(num_workers))
    start_time = time.time()
    list_cad_paths, list_obj_pose_paths, list_output_dirs = [], [], []
    for idx_obj, model_info in tqdm(enumerate(model_infos)):
        if idx_obj not in range(cfg.start_idx - 1, cfg.end_idx + 1):
            list_cad_paths.append(None)
            list_obj_pose_paths.append(None)
            list_output_dirs.append(None)

        else:
            obj_id = model_info["obj_id"]
            cat_id = model_info["shapenet_synset_id"]
            cad_id = model_info["shapenet_source_id"]
            cad_name = f"{cat_id}/{cad_id}/models/model_normalized_binormals.bam"
            cad_path = f"{cad_dir}/models_panda3d_bam/{cad_name}"

            obj_poses = template_poses.copy()
            obj_poses[:, :3, 3] *= 2
            object_center = np.array([0, 0, 0])
            obj_poses = re_pose_object(object_center, obj_poses)
            obj_pose_path = f"{root_save_dir}/object_poses/{obj_id:06d}.npy"
            np.save(obj_pose_path, obj_poses)

            # apply pitch=90 offset to object pose
            offset = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
            obj_poses = obj_poses @ offset

            obj_pose_with_offset_path = (
                f"{root_save_dir}/object_poses_with_offset/{obj_id:06d}.npy"
            )
            np.save(obj_pose_with_offset_path, obj_poses)
            output_dir = f"{root_save_dir}/{obj_id:06d}"

            list_cad_paths.append(cad_path)
            list_obj_pose_paths.append(obj_pose_with_offset_path)
            list_output_dirs.append(output_dir)

    list_cad_paths, list_obj_pose_paths, list_output_dirs = (
        np.array(list_cad_paths),
        np.array(list_obj_pose_paths),
        np.array(list_output_dirs),
    )

    cfg.end_idx = min(cfg.end_idx, len(list_cad_paths))
    list_cad_paths, list_obj_pose_paths, list_output_dirs = (
        list_cad_paths[cfg.start_idx : cfg.end_idx],
        list_obj_pose_paths[cfg.start_idx : cfg.end_idx],
        list_output_dirs[cfg.start_idx : cfg.end_idx],
    )

    logger.info("Start rendering for {} objects".format(len(list_cad_paths)))
    start_time = time.time()
    call_panda3d_with_index = partial(
        call_panda3d,
        list_cad_path=list_cad_paths,
        list_output_dir=list_output_dirs,
        list_obj_pose_path=list_obj_pose_paths,
        disable_output=True,
        num_gpus=num_gpus,
    )
    values = list(
        tqdm(
            pool.imap_unordered(call_panda3d_with_index, range(len(list_cad_paths))),
            total=len(list_cad_paths),
        )
    )
    correct_values = [val for val in values]
    logger.info(f"Finished for {len(correct_values)}/{len(list_cad_paths)} objects")
    finish_time = time.time()
    logger.info(f"Total time {len(list_cad_paths)}: {finish_time - start_time}")


if __name__ == "__main__":
    render()
