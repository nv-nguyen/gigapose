import logging
import os
import torchvision.transforms as T
from pathlib import Path
from src.utils.logging import get_logger
from src.custom_megapose.web_scene_dataset import (
    IterableWebSceneDataset,
    WebSceneDataset,
)
from tqdm import tqdm
from hydra.experimental import compose, initialize
from torch.utils.data import DataLoader

from src.megapose.datasets.object_dataset import RigidObjectDataset, RigidObject
from src.megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from src.megapose.panda3d_renderer.types import Panda3dLightData
from src.custom_megapose.transform import Transform


logger = get_logger(__name__)

if __name__ == "__main__":
    with initialize(config_path="../../../configs/"):
        cfg = compose(config_name="train.yaml")
    root_dir = Path(cfg.data.train.root_dir)
    save_dir = "./tmp/sample"
    os.makedirs(save_dir, exist_ok=True)
    wds_dir = root_dir / "gso"
    web_dataset = WebSceneDataset(wds_dir, load_depth=True, load_segmentation=True)
    web_dataloader = IterableWebSceneDataset(web_scene_dataset=web_dataset)

    # set up renderer and cads
    objects = []
    cad_dir = root_dir / "megapose/models/google_scanned_objects"
    for model_info in tqdm(cfg.data):
        cad_path = f"{cfg.data.train.gso_dataloader.cad.dir}/{cfg.data.train.gso_dataloader.cad.name}"
        cad_path = cad_path.replace("GSO_MODEL_ID", model_info["gso_id"])
        object = RigidObject(
            label=str(model_info["obj_id"]),
            mesh_path=cad_path,
            mesh_units="m",
            scaling_factor=0.1,
        )
        objects.append(object)
    rigid_object_dataset = RigidObjectDataset(objects)

    renderer = Panda3dSceneRenderer(rigid_object_dataset, verbose=False)
    # define light
    light_datas = [
        Panda3dLightData(
            light_type="ambient",
            color=((1.0, 1.0, 1.0, 1)),
        ),
    ]
    os.makedirs("./tmp", exist_ok=True)

    for scene_obs in web_dataloader:
        print(scene_obs)
