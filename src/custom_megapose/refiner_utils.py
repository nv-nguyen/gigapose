# Standard Library
from pathlib import Path
from typing import Union
import yaml
from omegaconf import OmegaConf
import torch
import os
import os.path as osp

from src.megapose.lib3d.rigid_mesh_database import MeshDataBase
from src.megapose.panda3d_renderer.panda3d_batch_renderer import Panda3dBatchRenderer
from src.megapose.training.pose_models_cfg import create_model_pose
from src.megapose.models.pose_rigid import PosePredictor
from src.megapose.inference.pose_estimator import PoseEstimator
from src.megapose.utils.load_model import NAMED_MODELS
from src.megapose.training.training_config import TrainingConfig
from src.megapose.utils.models_compat import change_keys_of_older_models
from src.megapose.training.pose_models_cfg import (
    check_update_config as check_update_config_pose,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def find_init_pose_path(save_dir, dataset_name, use_multiple):
    pred_dir = osp.join(save_dir, "predictions")
    files = [
        f for f in os.listdir(pred_dir) if f.endswith(".csv") and "refined" not in f
    ]
    files = sorted(files)
    logger.info(f"Found {len(files)} files in {pred_dir}")
    # assert len(files) == 2, f"More than 2 files found in {pred_dir}"
    if use_multiple:
        name_file = [f for f in files if "MultiHypothesis" in f]
        assert len(name_file) == 1, f"{len(name_file)} files found in {pred_dir}"
    else:
        name_file = [f for f in files if "MultiHypothesis" not in f]
        assert len(name_file) == 1, f"{len(name_file)} files found in {pred_dir}"
    name_file = name_file[0]
    logger.info(f"Using {name_file} as initial pose")
    assert dataset_name in name_file, f"{dataset_name} not in {name_file}"
    run_id = name_file.split(".")[0].split("_")[-1]
    model_name = name_file.split("-")[0]
    init_pose_path = osp.join(pred_dir, name_file)
    return init_pose_path, model_name, run_id


def load_cfg(path: Union[str, Path]) -> OmegaConf:
    cfg = yaml.load(Path(path).read_text(), Loader=yaml.UnsafeLoader)
    if isinstance(cfg, dict):
        cfg = OmegaConf.load(path)
    return cfg


def load_pretrained_refiner(cfg_refiner, object_dataset):
    model_info = NAMED_MODELS[cfg_refiner.model_name]
    mesh_database = MeshDataBase.from_object_ds(object_dataset)
    mesh_database_batched = mesh_database.batched()

    renderer_kwargs = {
        "preload_cache": False,
        "split_objects": False,
        "n_workers": cfg_refiner.num_workers,
    }
    models_root = Path(cfg_refiner.models_root)
    coarse_run_dir = models_root / model_info["coarse_run_id"]
    coarse_cfg: TrainingConfig = load_cfg(coarse_run_dir / "config.yaml")
    coarse_cfg = check_update_config_pose(coarse_cfg)

    refiner_run_dir = models_root / model_info["refiner_run_id"]
    refiner_cfg: TrainingConfig = load_cfg(refiner_run_dir / "config.yaml")
    refiner_cfg = check_update_config_pose(refiner_cfg)

    def make_renderer() -> Panda3dBatchRenderer:
        if renderer_kwargs is None:
            renderer_kwargs_ = dict()
        else:
            renderer_kwargs_ = renderer_kwargs
        renderer_kwargs_.setdefault("split_objects", True)
        renderer_kwargs_.setdefault("preload_cache", False)
        renderer_kwargs_.setdefault("n_workers", 4)
        renderer = Panda3dBatchRenderer(
            object_dataset=object_dataset, **renderer_kwargs_
        )
        return renderer

    def load_model(run_id: str, renderer: Panda3dBatchRenderer) -> PosePredictor:
        if run_id is None:
            return
        run_dir = models_root / run_id
        cfg: TrainingConfig = load_cfg(run_dir / "config.yaml")
        cfg = check_update_config_pose(cfg)
        model = create_model_pose(cfg, renderer=renderer, mesh_db=mesh_database_batched)
        ckpt = torch.load(run_dir / "checkpoint.pth.tar")
        ckpt = ckpt["state_dict"]
        ckpt = change_keys_of_older_models(ckpt)
        model.load_state_dict(ckpt)
        model.cfg = cfg
        model.config = cfg
        return model

    coarse_renderer = make_renderer()
    refiner_renderer = coarse_renderer

    coarse_model = load_model(model_info["coarse_run_id"], coarse_renderer)
    refiner_model = load_model(model_info["refiner_run_id"], refiner_renderer)
    pose_estimator = PoseEstimator(
        refiner_model=refiner_model,
        coarse_model=coarse_model,
        detector_model=None,
        depth_refiner=cfg_refiner.depth_refiner,
        bsz_objects=cfg_refiner.batch_size_objects,
        bsz_images=cfg_refiner.batch_size_images,
    )
    return pose_estimator
