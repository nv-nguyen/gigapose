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
    source_url = Path(
        "www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/megapose-models/"
    )
    model_names = ["coarse-rgb-906902141", "refiner-rgb-653307694"]
    files = ["checkpoint.pth.tar", "config.yaml"]
    for model_name in model_names:
        local_dir = root_dir / "pretrained/megapose-models/" / model_name
        os.makedirs(local_dir, exist_ok=True)

        for file in files:
            url = source_url / model_name / file
            download_cmd = f"wget -O {local_dir}/{file} {url}"
            logger.info(f"Running {download_cmd}")
            os.system(download_cmd)


if __name__ == "__main__":
    download()
