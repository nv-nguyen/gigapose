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
import glob

from src.lib3d.template_transform import get_obj_poses_from_template_level
from src.utils.logging import get_logger

logger = get_logger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../../configs",
    config_name="train",
)
def render(cfg) -> None:
    root_dir = Path(cfg.data.train.root_dir)
    model_infos = inout.load_json(root_dir / "shapenet/models_info.json")
    logger.info(f"Rendering {len(model_infos)} models!")

    root_save_dir = root_dir / "templates/shapenet"

    missing_ids = []
    for obj_id in tqdm(range(52472)):
        template_folder = root_save_dir / f"{obj_id:06d}"
        num_images = len(glob.glob(f"{template_folder}/*.png"))
        if num_images != 162 * 2:
            missing_ids.append(obj_id)
        if len(missing_ids) % 100 == 0 and len(missing_ids)>1:
            logger.info(f"Found {len(missing_ids)} missing templates")
    logger.info(f"Found {len(missing_ids)} missing templates")
    np.save("missing_ids.npy", np.array(missing_ids))


if __name__ == "__main__":
    render()
