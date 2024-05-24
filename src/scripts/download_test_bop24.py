from omegaconf import DictConfig, OmegaConf
import logging
import os
import os.path as osp
import hydra

# set level logging
logging.basicConfig(level=logging.INFO)


def run_download(config: DictConfig) -> None:
    local_dir = os.path.join(config.data.test.dataloader.root_dir, config.test_dataset_name)
    logging.info(f"Saving dataset to {local_dir}")

    download_cmd = f"huggingface-cli download bop-benchmark/datasets --include {config.test_dataset_name}/* --exclude *train_pbr* --local-dir {config.data.test.dataloader.root_dir} --repo-type=dataset"
    logging.info(f"Running {download_cmd}")
    # os.system(download_cmd)
    logging.info(f"Dataset downloaded to {local_dir}")

    # unzip the dataset
    os.system(f"export LOCAL_DIR={config.data.test.dataloader.root_dir}")
    os.system(f"export NAME={config.test_dataset_name}")
    unzip_cmd = "bash src/scripts/extract_bop.sh"
    logging.info(f"Running {unzip_cmd}")
    os.system(unzip_cmd)
    logging.info("Dataset extracted")


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="test",
)
def download(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    run_download(cfg)


if __name__ == "__main__":
    download()
