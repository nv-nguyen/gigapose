import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from src.utils.logging import get_logger
from src.utils.logging import start_disable_output, stop_disable_output
import wandb
import warnings

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="test")
def run_test(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    logger.info("Initializing logger, callbacks and trainer")
    cfg_trainer = cfg.machine.trainer
    if "WandbLogger" in cfg_trainer.logger._target_:
        os.environ["WANDB_API_KEY"] = cfg.user.wandb_api_key
        if cfg.machine.dryrun:
            os.environ["WANDB_MODE"] = "offline"
        logger.info(f"Wandb logger initialized at {cfg.save_dir}")
    elif "TensorBoardLogger" in cfg_trainer.logger._target_:
        tensorboard_dir = f"{cfg.save_dir}/{cfg_trainer.logger.name}"
        os.makedirs(tensorboard_dir, exist_ok=True)
        logger.info(f"Tensorboard logger initialized at {tensorboard_dir}")
    else:
        raise NotImplementedError("Only Wandb and Tensorboard loggers are supported")
    os.makedirs(cfg.save_dir, exist_ok=True)

    if cfg.disable_output:
        log = start_disable_output(os.path.join(cfg.save_dir, "test.log"))

    if cfg.machine.name == "slurm":
        num_gpus = int(os.environ["SLURM_GPUS_ON_NODE"])
        num_nodes = int(os.environ["SLURM_NNODES"])
        cfg_trainer.devices = num_gpus
        cfg_trainer.num_nodes = num_nodes
        logger.info(f"Slurm config: {num_gpus} gpus,  {num_nodes} nodes")

    trainer = instantiate(cfg_trainer)
    logger.info("Trainer initialized!")

    cfg.model.test_setting = cfg.test_setting
    model = instantiate(cfg.model)
    logger.info("Model initialized!")

    cfg.data.test.dataloader.dataset_name = cfg.test_dataset_name
    cfg.data.test.dataloader.batch_size = cfg.machine.batch_size
    cfg.data.test.dataloader.load_gt = False
    cfg.data.test.dataloader.test_setting = cfg.test_setting
    test_dataset = instantiate(cfg.data.test.dataloader)
    test_dataloader = DataLoader(
        test_dataset.web_dataloader.datapipeline,
        batch_size=1,  # a single image may have multiples instances
        num_workers=cfg.machine.num_workers,
        collate_fn=test_dataset.collate_fn,
    )

    # set template dataset as a part of the model
    cfg.data.test.dataloader.dataset_name = cfg.test_dataset_name
    cfg.data.test.dataloader._target_ = "src.dataloader.template.TemplateSet"
    template_dataset = instantiate(cfg.data.test.dataloader)

    model.template_datasets = {cfg.test_dataset_name: template_dataset}
    model.test_dataset_name = cfg.test_dataset_name
    model.max_num_dets_per_forward = cfg.max_num_dets_per_forward
    if cfg.run_id is None:
        model.run_id = wandb.run.id
    else:
        model.run_id = cfg.run_id
    model.log_interval = len(test_dataloader) // 30
    logger.info("Dataloaders initialized!")

    trainer.test(
        model, dataloaders=test_dataloader, ckpt_path=cfg.model.checkpoint_path
    )

    if cfg.disable_output:
        stop_disable_output(log)
    logger.info("Done!")


if __name__ == "__main__":
    run_test()
