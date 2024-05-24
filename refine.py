# Standard Library
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from megapose.datasets.bop_object_datasets import BOPObjectDataset
from megapose.utils.logging import get_logger
from src.custom_megapose.refiner_utils import find_init_pose_path
from src.models.refiner import Refiner
import time
from src.utils.gpu import assign_gpu, terminate_processes
import warnings

warnings.filterwarnings("ignore")
logger = get_logger(__name__)
assign_gpu()


@hydra.main(version_base=None, config_path="configs", config_name="test")
def run_refiner(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    logger.info("Loading dataset ...")
    init_loc_path, model_name, run_id = find_init_pose_path(
        cfg.save_dir, cfg.test_dataset_name, cfg.use_multiple
    )
    cfg.data.test.dataloader.batch_size = 1
    cfg.data.test.dataloader.dataset_name = cfg.test_dataset_name
    cfg.data.test.dataloader.init_loc_path = init_loc_path
    cfg.data.test.dataloader.test_setting = cfg.test_setting
    test_dataset = instantiate(cfg.data.test.dataloader)

    dataloader = DataLoader(
        test_dataset.web_dataloader.datapipeline,
        batch_size=1,  # cfg.machine.batch_size
        num_workers=10,
        collate_fn=test_dataset.collate_refine_fn,
    )
    logger.info(f"Prediction is from Model={model_name}, run_id={run_id} done!")

    # load cad models for refinement
    root_dir = Path(cfg.data.test.dataloader.root_dir)
    cad_dir = root_dir / cfg.test_dataset_name / "models"
    if cfg.test_dataset_name == "tless":
        cad_dir = str(cad_dir) + "_cad"
    object_dataset = BOPObjectDataset(
        Path(cad_dir), format=".obj" if "Wonder3d" in cfg.test_dataset_name else ".ply"
    )
    logger.info("Loading CAD dataset done!")

    refiner = Refiner(
        object_dataset=object_dataset,
        cfg_refiner_model=cfg.model.refiner,
        use_multiple=cfg.use_multiple,
        log_dir=cfg.save_dir,
        test_dataset_name=cfg.test_dataset_name,
        coarse_model_name=model_name,
        run_id=run_id,
    )

    limit_test_batches = None
    if limit_test_batches is not None:
        logger.info(f"Limiting test batches to {limit_test_batches}")
        cfg.machine.trainer.limit_test_batches = limit_test_batches
    trainer = instantiate(cfg.machine.trainer)
    logger.info("Trainer initialized!")

    logger.info("Refining poses ...")
    start_time = time.time()
    trainer.test(refiner, dataloaders=dataloader)
    run_time = time.time() - start_time
    logger.info(f"Refining poses done in {run_time}!")
    terminate_processes()


if __name__ == "__main__":
    run_refiner()
