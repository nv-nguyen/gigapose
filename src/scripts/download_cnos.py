import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.logging import get_logger

logger = get_logger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train",
)
def download_cnos(cfg: DictConfig) -> None:
    cfg_data = cfg.data.test
    cfg_data.root_dir = Path(cfg_data.root_dir)
    tmp_dir = cfg_data.root_dir / "tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    OmegaConf.set_struct(cfg, False)

    download_cmd = (
        f"wget -O {tmp_dir}/cnos.zip {cfg_data.source_cnos_url} --no-check-certificate"
    )
    logger.info(f"Running {download_cmd}")
    os.system(download_cmd)

    zip_path = tmp_dir / "cnos.zip"
    unzip_cmd = f"unzip {zip_path} -d {tmp_dir}"
    logger.info(f"Running {unzip_cmd}")
    os.system(unzip_cmd)

    # rename
    os.rename(
        tmp_dir / "bop23_default_detections_for_task4/cnos-fastsam",
        cfg_data.root_dir / "cnos-fastsam",
    )
    logger.info(
        f"All CNOS's detections are available at {cfg_data.root_dir}/cnos-fastsam !"
    )


if __name__ == "__main__":
    download_cnos()
