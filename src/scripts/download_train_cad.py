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
    cfg_data = cfg.data.train
    cfg_data.root_dir = Path(cfg_data.root_dir)

    tmp_dir = cfg_data.root_dir / "tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    source_url = "https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/tars/"
    metaData = {"gso": "google_scanned_objects", "shapenet": "shapenetcorev2"}

    for dataset_name in ["shapenet"]:
        local_dir = cfg_data.root_dir / f"{dataset_name}" / "models"
        os.makedirs(local_dir, exist_ok=True)

        zip_name = metaData[dataset_name]
        url = f"{source_url}{zip_name}.zip"
        models_cmd = f"wget -O {tmp_dir}/{zip_name}.zip {url} --no-check-certificate"
        logger.info(f"Running {models_cmd}")
        os.system(models_cmd)

        unzip_cmd = f"unzip {tmp_dir}/{zip_name}.zip -d {tmp_dir}"
        logger.info(f"Running {unzip_cmd}")
        os.system(unzip_cmd)

        rename_cmd = f"mv {tmp_dir}/{zip_name}/* {local_dir}"
        logger.info(f"Running {rename_cmd}")
        os.system(rename_cmd)


if __name__ == "__main__":
    download()
