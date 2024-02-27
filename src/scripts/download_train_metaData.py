import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_download(cfg_dataset: DictConfig) -> None:
    os.makedirs(cfg_dataset.local_dir, exist_ok=True)

    models_info_cmd = f"wget -O {cfg_dataset.models_info_path} {cfg_dataset.models_info_url} --no-check-certificate"
    logger.info(f"Running {models_info_cmd}")
    os.system(models_info_cmd)

    key_to_shard_cmd = f"wget -O {cfg_dataset.key_to_shard_path} {cfg_dataset.key_to_shard_url} --no-check-certificate"
    logger.info(f"Running {key_to_shard_cmd}")
    os.system(key_to_shard_cmd)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train",
)
def download(cfg: DictConfig) -> None:
    cfg_data = cfg.data.train
    cfg_data.root_dir = Path(cfg_data.root_dir)

    OmegaConf.set_struct(cfg, False)
    for dataset_name in [
        "gso",
        "shapenet",
    ]:
        logger.info(f"Downloading {dataset_name}")
        models_info_url = f"{cfg_data.source_url}/bop23_datasets/megapose-{dataset_name}/{dataset_name}_models.json"
        models_info_path = (
            cfg_data.root_dir / dataset_name / "models_info.json"
        )
        key_to_shard_url = f"{cfg_data.source_url}/bop23_datasets/megapose-{dataset_name}/train_pbr_web/key_to_shard.json"
        key_to_shard_path = (
            cfg_data.root_dir / dataset_name / "key_to_shard.json"
        )
        cfg_dataset = OmegaConf.create(
            {
                "name": dataset_name,
                "models_info_url": models_info_url,
                "key_to_shard_url": key_to_shard_url,
                "local_dir": cfg_data.root_dir / dataset_name,
                "models_info_path": models_info_path,
                "key_to_shard_path": key_to_shard_path,
            }
        )
        run_download(cfg_dataset)
        logger.info("---" * 100)


if __name__ == "__main__":
    download()
