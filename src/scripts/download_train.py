import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_download(cfg_dataset: DictConfig) -> None:
    os.makedirs(cfg_dataset.local_dir / "train_pbr_web", exist_ok=True)
    for shard_id in range(0, 1040):
        source_path = cfg_dataset.url + f"shard-{shard_id:06d}.tar"
        target_path = (
            cfg_dataset.local_dir / "train_pbr_web" / f"shard-{shard_id:06d}.tar"
        )
        download_cmd = f"wget -O {target_path} {source_path} --no-check-certificate"
        logger.info(f"Running {download_cmd}")
        os.system(download_cmd)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train",
)
def download(cfg: DictConfig) -> None:
    cfg_data = cfg.data.train
    cfg_data.root_dir = Path(cfg_data.root_dir)
    tmp_dir = cfg_data.root_dir / "tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    OmegaConf.set_struct(cfg, False)
    for dataset_name in [
        "gso",
        "shapenet",
    ]:
        logger.info(f"Downloading {dataset_name}")
        url = f"{cfg_data.source_url}/bop23_datasets/megapose-{dataset_name}/train_pbr_web/"
        cfg_dataset = OmegaConf.create(
            {
                "name": dataset_name,
                "url": url,
                "local_dir": cfg_data.root_dir / dataset_name,
                "tmp": tmp_dir,
            }
        )
        run_download(cfg_dataset)
        logger.info("---" * 100)


if __name__ == "__main__":
    download()
