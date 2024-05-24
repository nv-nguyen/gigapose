import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_download(cfg_dataset: DictConfig) -> None:
    os.makedirs(cfg_dataset.local_dir, exist_ok=True)
    for key in ["rgb", "cad", "base"]:
        tmp_dir = cfg_dataset.tmp / f"{cfg_dataset.name}_{key}.zip"
        download_cmd = f"wget -O {tmp_dir} {cfg_dataset[key]} --no-check-certificate"
        logger.info(f"Running {download_cmd}")
        os.system(download_cmd)

        unzip_cmd = f"unzip {tmp_dir} -d {cfg_dataset.local_dir}"
        if key == "base":
            unzip_cmd = unzip_cmd.replace("unzip", "unzip -j")
        logger.info(f"Running {unzip_cmd}")
        os.system(unzip_cmd)

    # formatting the dataset top webdataset format
    imagewise_cmd = "python -m src.scripts.convert_scenewise_to_imagewise --input INPUT --output OUTPUT --nprocs 10"
    webdataset_cmd = "python -m src.scripts.convert_imagewise_to_webdataset --input INPUT --output OUTPUT"

    if cfg_dataset.name in ["tless", "hb"]:
        split = "test_primesense"
    else:
        split = "test"
    split_dir = cfg_dataset.local_dir / split
    tmp_dir = cfg_dataset.tmp / f"{cfg_dataset.name}_image_wise" / split
    os.makedirs(tmp_dir, exist_ok=True)

    imagewise_cmd = imagewise_cmd.replace("INPUT", str(split_dir))
    imagewise_cmd = imagewise_cmd.replace("OUTPUT", str(tmp_dir))

    logger.info(f"Running {imagewise_cmd}")
    os.system(imagewise_cmd)

    webdataset_cmd = webdataset_cmd.replace("INPUT", str(tmp_dir))
    webdataset_cmd = webdataset_cmd.replace("OUTPUT", str(split_dir))

    if cfg_dataset.name in ["tless", "ycbv", "tudl", "itodd"]:
        webdataset_cmd += " --nprocs 4"
    logger.info(f"Running {webdataset_cmd}")
    os.system(webdataset_cmd)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train",
)
def download(cfg: DictConfig) -> None:
    cfg_data = cfg.data.test
    cfg_data.root_dir = Path(cfg_data.root_dir)
    tmp_dir = cfg_data.root_dir / "tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    OmegaConf.set_struct(cfg, False)
    for dataset_name in [
        "lmo",
        "tless",
        "tudl",
        "icbin",
        "itodd",  # gt not available
        "hb",  # gt not available
        "ycbv",
    ]:
        logger.info(f"Downloading {dataset_name}")
        if dataset_name in ["tless", "hb"]:
            prefix = "_primesense"
        else:
            prefix = ""
        rgb_url = f"{cfg_data.source_url}/{dataset_name}_test{prefix}_bop19.zip"
        cad_url = f"{cfg_data.source_url}/{dataset_name}_models.zip"
        base_url = f"{cfg_data.source_url}/{dataset_name}_base.zip"
        cfg_dataset = OmegaConf.create(
            {
                "name": dataset_name,
                "rgb": rgb_url,
                "cad": cad_url,
                "base": base_url,
                "local_dir": cfg_data.root_dir / dataset_name,
                "tmp": tmp_dir,
            }
        )
        run_download(cfg_dataset)
        logger.info("---" * 100)


if __name__ == "__main__":
    download()
