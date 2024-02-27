import os
from pathlib import Path
import hydra
from omegaconf import DictConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train",
)
def download(cfg: DictConfig) -> None:
    root_dir = Path(cfg.machine.root_dir)
    source_url = (
        "https://huggingface.co/datasets/nv-nguyen/gigaPose/resolve/main/templates.zip"
    )
    tmp_dir = root_dir / "datasets/tmp/"
    os.makedirs(tmp_dir, exist_ok=True)

    download_cmd = f"wget -O {tmp_dir}/templates.zip {source_url}"
    logger.info(f"Running {download_cmd}")
    # os.system(download_cmd)

    unzip_cmd = f"unzip {tmp_dir}/templates.zip -d {tmp_dir}"
    logger.info(f"Running {unzip_cmd}")
    os.system(unzip_cmd)
    
    os.rename(
        tmp_dir / "templates",
        root_dir / "datasets/templates",
    )

if __name__ == "__main__":
    download()
