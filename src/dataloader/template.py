from __future__ import annotations

# Standard Library
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union

# Third Party
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
from bop_toolkit_lib import inout, pycoco_utils
from src.utils.dataset import LMO_index_to_ID, LMO_ID_to_index
from src.custom_megapose.template_dataset import TemplateDataset, NearestTemplateFinder
import torch
from src.utils.bbox import BoundingBox
import src.megapose.utils.tensor_collection as tc


class TemplateSet(Dataset):
    def __init__(
        self,
        root_dir,
        dataset_name,
        template_config,
        transforms,
        **kwargs,
    ):
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.transforms = transforms

        # load the dataset
        cad_name = self.get_cad_name(dataset_name)

        # load the template dataset
        model_infos = inout.load_json(
            self.root_dir / self.dataset_name / cad_name / "models_info.json"
        )
        self.model_infos = [{"obj_id": int(obj_id)} for obj_id in model_infos.keys()]

        template_config.dir += f"/{dataset_name}"
        self.template_dataset = TemplateDataset.from_config(
            self.model_infos, template_config
        )
        self.template_finder = NearestTemplateFinder(template_config)

    def get_cad_name(self, dataset_name):
        if dataset_name in ["tless"]:
            cad_name = "models_cad"
        else:
            cad_name = "models"
        return cad_name

    def __len__(self):
        return len(self.model_infos)

    def __getitem__(self, index):
        # loading templates
        if "lmo" in self.dataset_name:
            label = LMO_index_to_ID[index]
        else:
            label = f"{index+1}"

        # load template data
        template_data = self.template_dataset.get_object_templates(label)
        data, poses = template_data.read_test_mode()

        # crop the template
        cropped_data = self.transforms.crop_transform(data["box"], images=data["rgba"])
        cropped_data["images"][:, :3] = self.transforms.normalize(
            cropped_data["images"][:, :3]
        )
        data["K"] = torch.from_numpy(self.template_dataset.K).float()

        out_data = tc.PandasTensorCollection(
            K=data["K"],
            rgb=cropped_data["images"][:, :3],
            mask=cropped_data["images"][:, -1],
            M=cropped_data["M"],
            poses=poses,
            infos=pd.DataFrame(),
        )
        return out_data


if __name__ == "__main__":
    from tqdm import tqdm
    from hydra.experimental import compose, initialize
    from hydra.utils import instantiate
    from omegaconf import DictConfig, OmegaConf
    from src.libVis.torch import inv_rgb_transform
    from torchvision.utils import save_image

    with initialize(config_path="../../configs/"):
        cfg = compose(config_name="test.yaml")
    OmegaConf.set_struct(cfg, False)

    save_dir = "./tmp"
    os.makedirs(save_dir, exist_ok=True)

    cfg.machine.batch_size = 9
    cfg.data.test.dataloader.dataset_name = "ycbv"
    cfg.data.test.dataloader._target_ = "src.dataloader.template.TemplateSet"
    template_dataset = instantiate(cfg.data.test.dataloader)
    for idx_batch in tqdm(range(len(template_dataset))):
        data = template_dataset[idx_batch]
        templates = data.rgb
        templates = inv_rgb_transform(templates)
        save_image(
            templates,
            os.path.join(save_dir, f"{idx_batch:06d}.png"),
            nrow=16,
        )
