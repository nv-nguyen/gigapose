# adapted from https://github.com/megapose6d/megapose6d/blob/master/src/megapose/datasets/web_scene_dataset.py#L1
import io
import json
import tarfile
from functools import partial
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

# Third Party
import imageio
import numpy as np
import pandas as pd
import webdataset as wds
from bop_toolkit_lib import inout, pycoco_utils
from src.megapose.lib3d.transform import Transform
from src.utils.logging import get_logger
from src.megapose.utils.webdataset import tarfile_to_samples
from src.megapose.datasets.scene_dataset import (
    CameraData,
    IterableSceneDataset,
    ObjectData,
    ObservationInfos,
    SceneDataset,
    SceneObservation,
)

logger = get_logger(__name__)


def load_scene_ds_obs(
    sample: Dict[str, Union[bytes, str]],
    depth_scale: float = 1.0,
    load_depth: bool = False,
    label_format: str = "{label}",
) -> SceneObservation:
    # load rgb
    depth_format = "depth.png"
    if "rgb.jpg" in sample:
        rgb = np.array(imageio.imread(io.BytesIO(sample["rgb.jpg"])))
    elif "rgb.png" in sample:
        rgb = np.array(imageio.imread(io.BytesIO(sample["rgb.png"])))
    elif "gray.tif" in sample:
        rgb = np.array(imageio.imread(io.BytesIO(sample["gray.tif"])))
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
        depth_format = "depth.tif"
    else:
        raise ValueError("No rgb image found")

    if "gt.json" in sample:
        gt_available = True
    else:
        gt_available = False

    depth = None
    if load_depth:
        depth = imageio.imread(io.BytesIO(sample[depth_format]))
        depth = np.asarray(depth, dtype=np.float32)
        depth /= depth_scale

    camera_data = json.loads(sample["camera.json"])
    if "cam_R_w2c" in camera_data:
        cam_R_w2c = np.array(camera_data["cam_R_w2c"]).reshape(3, 3)
        cam_t_w2c = np.array(camera_data["cam_t_w2c"])
    else:
        cam_R_w2c = np.eye(3)
        cam_t_w2c = np.zeros(3)

    camera_data = CameraData(
        K=np.array(camera_data["cam_K"]).reshape(3, 3),
        TWC=Transform(
            cam_R_w2c,
            cam_t_w2c,
        ),
        resolution=rgb.shape[:2],
    )

    scene_id, view_id = sample["__key__"].split("_")
    infos = ObservationInfos(scene_id=scene_id, view_id=view_id)

    if gt_available:
        objects_gt = json.loads(sample["gt.json"])
        objects_gt_info = json.loads(sample["gt_info.json"])
        object_datas = []
        for idx_obj, data in enumerate(objects_gt):
            data_info = objects_gt_info[idx_obj]
            data["visib_fract"] = data_info["visib_fract"]
            if data["visib_fract"] <= 0.1:
                continue
            data["bbox_modal"] = data_info["bbox_visib"]
            data["bbox_amodal"] = data_info["bbox_obj"]

            obj_id = data["obj_id"]
            data["label"] = f"{obj_id}"
            data["unique_id"] = f"{idx_obj}"
            data["TWO"] = [data["cam_R_m2c"], data["cam_t_m2c"]]

            object_data = ObjectData.from_json(data)
            object_datas.append(object_data)

        masks_visib_ = json.loads(sample["mask_visib.json"])
        masks_visib = {
            k: pycoco_utils.rle_to_binary_mask(v) for k, v in masks_visib_.items()
        }
    else:
        object_datas = []
        masks_visib = {}
    return SceneObservation(
        rgb=rgb,
        depth=depth,
        infos=infos,
        object_datas=object_datas,
        camera_data=camera_data,
        binary_masks=masks_visib,
    )


class WebSceneDataset(SceneDataset):
    def __init__(
        self,
        wds_dir: Path,
        depth_scale: float = 1000.0,
        load_depth: bool = True,
        load_segmentation: bool = True,
        label_format: str = "{label}",
        load_frame_index: bool = False,
    ):
        self.depth_scale = depth_scale
        self.label_format = label_format
        self.wds_dir = wds_dir

        frame_index = self.load_frame_index()

        super().__init__(
            frame_index=frame_index,
            load_depth=load_depth,
            load_segmentation=load_segmentation,
        )

    def load_frame_index(self) -> pd.DataFrame:
        if "test" in self.wds_dir.name:
            root_dir = self.wds_dir
        else:
            root_dir = self.wds_dir.parent
        key_to_shard = inout.load_json(root_dir / "key_to_shard.json")
        keys, shard_fnames = [], []
        for key, shard_fname in key_to_shard.items():
            keys.append(key)
            shard_fnames.append(shard_fname)
        frame_index = pd.DataFrame({"key": keys, "shard_fname": shard_fnames})
        return frame_index

    def get_tar_list(self) -> List[str]:
        # broken shards of MegaPose-GSO and MegaPose-ShapeNet
        invalid_shards = [101, 102, 108, 114, 701]
        tar_files = [
            str(x)
            for x in self.wds_dir.iterdir()
            if x.suffix == ".tar" and int(x.stem.split("-")[1]) not in invalid_shards
        ]
        tar_files.sort()
        logger.info(f"WebSceneDataset: {len(tar_files)} shards")
        return tar_files

    def __getitem__(self, idx: int) -> SceneObservation:
        assert self.frame_index is not None
        row = self.frame_index.iloc[idx]
        shard_fname, key = row.shard_fname, row.key
        tar = tarfile.open(self.wds_dir / shard_fname)

        sample: Dict[str, Union[bytes, str]] = dict()
        for k in (
            "rgb.png",
            "segmentation.png",
            "depth.png",
            "infos.json",
            "object_datas.json",
            "camera_data.json",
        ):
            tar_file = tar.extractfile(f"{key}.{k}")
            assert tar_file is not None
            sample[k] = tar_file.read()

        obs = load_scene_ds_obs(sample, load_depth=self.load_depth)
        tar.close()
        return obs


class IterableWebSceneDataset(IterableSceneDataset):
    def __init__(
        self, web_scene_dataset: WebSceneDataset, set_length: Optional[int] = True
    ):
        self.web_scene_dataset = web_scene_dataset

        load_scene_ds_obs_ = partial(
            load_scene_ds_obs,
            depth_scale=self.web_scene_dataset.depth_scale,
            load_depth=self.web_scene_dataset.load_depth,
            label_format=self.web_scene_dataset.label_format,
        )

        def load_scene_ds_obs_iterator(
            samples: Iterator[SceneObservation],
        ) -> Iterator[SceneObservation]:
            for sample in samples:
                yield load_scene_ds_obs_(sample)

        self.datapipeline = wds.DataPipeline(
            wds.SimpleShardList(
                self.web_scene_dataset.get_tar_list()
            ),  # wds.ResampledShards(),
            wds.split_by_worker,
            tarfile_to_samples(),
            load_scene_ds_obs_iterator,
            # wds.shuffle(buffer_size),
        )
        if set_length:
            length = len(self.web_scene_dataset)
            self.datapipeline.with_length(length)
            logger.info(f"IterableWebSceneDataset: {length} samples")

    def __iter__(self) -> Iterator[SceneObservation]:
        return iter(self.datapipeline)
