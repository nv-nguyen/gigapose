import os
import wandb

os.environ["MPLCONFIGDIR"] = os.getcwd() + "./tmp/"
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from src.utils.weight import load_checkpoint
from src.utils.logging import get_logger
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
import warnings

warnings.filterwarnings("ignore")
pl.seed_everything(2023)
logger = get_logger(__name__)

train_dataset_names = {0: ["gso"], 1: ["shapenet"], 2: ["gso", "shapenet"]}


@hydra.main(version_base=None, config_path="configs", config_name="train")
def run_train(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    logger.info(f"Checkpoints will be stored in: {cfg.callback.checkpoint.dirpath}")

    logger.info("Initializing logger, callbacks and trainer")
    os.environ["WANDB_API_KEY"] = cfg.user.wandb_api_key
    if cfg.machine.dryrun:
        os.environ["WANDB_MODE"] = "offline"
    logger.info(f"Wandb logger initialized at {cfg.save_dir}")
    os.makedirs(cfg.save_dir, exist_ok=True)

    if cfg.machine.name == "slurm":
        num_gpus = int(os.environ["SLURM_GPUS_ON_NODE"])
        num_nodes = int(os.environ["SLURM_NNODES"])
        cfg.machine.trainer.devices = num_gpus
        cfg.machine.trainer.num_nodes = num_nodes
        logger.info(f"Slurm config: {num_gpus} gpus,  {num_nodes} nodes")
    # cfg.machine.trainer.limit_val_batches = 20
    # cfg.machine.trainer.num_sanity_val_steps = 10

    cfg.machine.trainer.check_val_every_n_epoch = None
    trainer = instantiate(cfg.machine.trainer)

    logger.info("Initializing dataloader")
    cfg.data.train.dataloader.batch_size = cfg.machine.batch_size
    cfg.data.test.dataloader.batch_size = cfg.machine.batch_size

    selected_train_dataset_names = train_dataset_names[cfg.train_dataset_id]
    train_dataloaders = []
    for name in selected_train_dataset_names:
        cfg.data.train.dataloader.dataset_name = name
        train_dataset = instantiate(cfg.data.train.dataloader)
        train_dataloader = DataLoader(
            train_dataset.web_dataloader.datapipeline,
            batch_size=cfg.machine.batch_size,
            num_workers=cfg.machine.num_workers,
            collate_fn=train_dataset.collate_fn,
        )
        train_dataloaders.append(train_dataloader)
    logger.info(f"Using {selected_train_dataset_names} datasets")

    cfg.data.test.dataloader.dataset_name = "ycbv"
    val_dataset = instantiate(cfg.data.test.dataloader)
    val_dataloader = DataLoader(
        val_dataset.web_dataloader.datapipeline,
        batch_size=1,  # a single image may have multiples instances
        num_workers=cfg.machine.num_workers,
        collate_fn=val_dataset.collate_fn,
    )

    logger.info("Initializing model")
    model = instantiate(cfg.model)

    # if model is resnet, load pretrained weight
    if cfg.model.ist_net.pretrained_weights is not None and cfg.use_pretrained:
        load_checkpoint(
            model.ist_net,
            cfg.model.ist_net.pretrained_weights,
            checkpoint_key=cfg.model.ist_net.checkpoint_key,
            prefix="",
        )

    logger.info(f"Starting training")
    trainer.fit(
        model,
        train_dataloaders=train_dataloaders,
        val_dataloaders=val_dataloader,
        # ckpt_path=cfg.model.checkpoint_path
        # if cfg.model.checkpoint_path is not None and cfg.use_pretrained
        # else None,
    )
    logger.info(f"---" * 20)


if __name__ == "__main__":
    run_train()
