from __future__ import annotations

# Standard Library
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Third Party
import numpy as np
import torch
from pathlib import Path

# MegaPose
import src.megapose.utils.tensor_collection as tc
from src.megapose.utils.tensor_collection import PandasTensorCollection
from src.megapose.datasets.scene_dataset import (
    SceneObservation,
)
from src.custom_megapose.web_scene_dataset import (
    WebSceneDataset,
)
from src.custom_megapose.web_scene_dataset import IterableWebSceneDataset
from src.utils.bbox import BoundingBox
from src.custom_megapose.transform import Transform
from src.custom_megapose.template_dataset import TemplateDataset, NearestTemplateFinder
from src.dataloader.keypoints import KeypointInput, KeyPointSampler
from src.utils.logging import get_logger
from src.lib3d.torch import get_relative_scale_inplane
from bop_toolkit_lib import inout
from torchvision.utils import save_image

ListBbox = List[int]
ListPose = List[List[float]]
SceneObservationTensorCollection = PandasTensorCollection

SingleDataJsonType = Union[str, float, ListPose, int, ListBbox, Any]
DataJsonType = Union[Dict[str, SingleDataJsonType], List[SingleDataJsonType]]

logger = get_logger(__name__)


@dataclass
class GigaPoseTrainSet:
    def __init__(
        self,
        batch_size,
        root_dir,
        dataset_name,
        depth_scale,
        template_config,
        transforms,
    ):
        self.batch_size = batch_size
        self.dataset_dir = Path(root_dir) / dataset_name
        self.transforms = transforms
        if self.transforms.rgb_augmentation:
            self.transforms.rgb_transform.transform = [
                transform for transform in self.transforms.rgb_transform.transform
            ]

        # load the dataset
        web_dataset = WebSceneDataset(
            self.dataset_dir / "train_pbr_web", depth_scale=depth_scale
        )
        self.web_dataloader = IterableWebSceneDataset(web_dataset)

        # load the template dataset
        model_infos = inout.load_json(self.dataset_dir / "models_info.json")

        template_config.dir += f"/{dataset_name}"
        self.template_dataset = TemplateDataset.from_config(
            model_infos, template_config
        )
        self.template_finder = NearestTemplateFinder(template_config)

        # keypoint sampler
        self.keypoint_sampler = KeyPointSampler()

    def process_real(self, batch, test_mode=False, min_box_size=50):
        # get the ground truth data
        rgb = batch["rgb"] / 255.0
        depth = batch["depth"].squeeze(1)
        detections = batch["gt_detections"]
        data = batch["gt_data"]

        # make bounding box square
        bboxes = BoundingBox(detections.bboxes, "xywh")
        if test_mode:
            idx_selected = np.arange(len(detections.bboxes))
        else:
            # keep only valid bounding boxes
            idx_selected = np.random.choice(
                np.arange(len(detections.bboxes)),
                min(self.batch_size, len(detections.bboxes)),
                replace=False,
            )

        bboxes = bboxes.reset(idx_selected)
        batch_im_id = detections[idx_selected].infos.batch_im_id
        masks = data.masks[idx_selected]
        K = data.K[idx_selected].float()
        pose = data.TWO[idx_selected].float()

        rgb = rgb[batch_im_id]
        depth = depth[batch_im_id]
        m_rgb = rgb * masks[:, None, :, :]

        m_rgba = torch.cat([m_rgb, masks[:, None, :, :]], dim=1)
        cropped_data = self.transforms.crop_transform(bboxes.xyxy_box, images=m_rgba)

        out_data = tc.PandasTensorCollection(
            full_rgb=rgb,
            full_depth=depth,
            K=K,
            rgb=cropped_data["images"][:, :3],
            mask=cropped_data["images"][:, -1],
            M=cropped_data["M"],
            pose=pose,
            infos=data[idx_selected].infos,
        )

        return out_data

    def process_template(self, real_data: SceneObservationTensorCollection):
        names = [
            "full_rgba",
            "full_depth",
            "K",
            "bboxes",
            "pose",
            "T_real2template",
            "T_template2real",
        ]
        template_data = {name: [] for name in names}

        labels = real_data.infos.label
        real_poses = real_data.pose.cpu().numpy()
        K = torch.from_numpy(self.template_dataset.K).float()
        for label, real_pose in zip(labels, real_poses):
            # find nearest etmplate
            info = self.template_finder.search_nearest_template(real_pose[:3, :3])
            if self.transforms.inplane_augmentation:
                inplane = np.random.randint(0, 360)
                inplane_transform = Transform.from_inplane(inplane)
                info["inplane"] = inplane
            else:
                info["inplane"] = 0

            # load template data
            template_object = self.template_dataset.get_object_templates(label)
            data = template_object.read_train_mode(
                transform=None,
                object_info=info,
                num_negatives=0,
            )
            if (
                self.transforms.inplane_augmentation
            ):  # apply inplane augmentation for positive
                data["pos_pose"] = inplane_transform * data["pos_pose"]

            # compute relative pose template -> real, real -> template
            real_pose = Transform(real_pose)
            relPose = data["pos_pose"] * real_pose.inverse()
            relPose_inv = relPose.inverse()

            template_data["full_rgba"].append(data["pos_rgba"])
            template_data["full_depth"].append(data["pos_depth"])
            template_data["K"].append(K)
            template_data["bboxes"].append(data["pos_box"])

            template_data["T_real2template"].append(relPose.toTensor())
            template_data["T_template2real"].append(relPose_inv.toTensor())

            template_pose = data["pos_pose"].toTensor()
            template_init_scale = torch.norm(template_pose[:3, 0])
            if np.abs(template_init_scale - 1) > 1e-3:
                template_pose[:3, :3] /= template_init_scale
                template_pose[:3, 3] /= template_init_scale
            template_data["pose"].append(template_pose)

        # stack the data
        for name in names:
            template_data[name] = torch.stack(template_data[name], dim=0)

        # crop the template
        cropped_data = self.transforms.crop_transform(
            template_data["bboxes"], images=template_data["full_rgba"]
        )

        out_data = tc.PandasTensorCollection(
            full_rgb=template_data["full_rgba"],
            full_depth=template_data["full_depth"].squeeze(1),
            K=template_data["K"],
            rgb=cropped_data["images"][:, :3],
            mask=cropped_data["images"][:, -1],
            M=cropped_data["M"],
            pose=template_data["pose"],
            infos=real_data.infos,
        )
        return (
            out_data,
            template_data["T_real2template"],
            template_data["T_template2real"],
        )

    def process_keypoints(
        self, real_data, template_data, T_real2template, T_template2real
    ):
        all_data = {}
        for name, data in zip(["real", "template"], [real_data, template_data]):
            all_data[name] = KeypointInput(
                full_rgb=data.full_rgb,
                full_depth=data.full_depth,
                K=data.K,
                M=data.M,
                mask=data.mask,
                rgb=data.rgb,
            )
        # compute keypoints from template to real
        keypoint_data = self.keypoint_sampler.sample_pts(
            src_data=all_data["template"],
            tar_data=all_data["real"],
            T_src2target=T_template2real,
            T_tar2source=T_real2template,
        )

        # compute gt relScale, relInplane template -> real
        relScale, relInplane = get_relative_scale_inplane(
            src_K=template_data.K,
            tar_K=real_data.K,
            src_pose=template_data.pose,
            tar_pose=real_data.pose,
            src_M=template_data.M,
            tar_M=real_data.M,
        )
        return keypoint_data, dict(relScale=relScale, relInplane=relInplane)

    def collate_fn(
        self,
        batch: List[SceneObservation],
    ):
        try:
            if self.transforms.rgb_augmentation:
                batch = [self.transforms.rgb_transform(data) for data in batch]
            # convert to tensor collection
            batch = SceneObservation.collate_fn(batch)

            # load real data
            real_data = self.process_real(batch)

            # load template data
            template_data, T_real2temp, T_temp2real = self.process_template(real_data)

            # compute keypoints, template is source, real is target
            keypoints, rel_data = self.process_keypoints(
                real_data, template_data, T_real2temp, T_temp2real
            )

            out_data = tc.PandasTensorCollection(
                src_img=self.transforms.normalize(template_data.rgb),
                src_mask=template_data.mask,
                src_K=template_data.K,
                src_M=template_data.M,
                src_pts=keypoints["src_pts"],
                tar_img=self.transforms.normalize(real_data.rgb),
                tar_mask=real_data.mask,
                tar_K=real_data.K,
                tar_M=real_data.M,
                tar_pts=keypoints["tar_pts"],
                relScale=rel_data["relScale"],
                relInplane=rel_data["relInplane"],
                infos=real_data.infos,
            )
        except Exception as e:
            logger.info(f"Error {e}")
            return None

        return out_data


if __name__ == "__main__":
    import time

    from hydra.experimental import compose, initialize
    from src.utils.dataloader import NoneFilteringDataLoader
    from src.megapose.datasets.scene_dataset import SceneObservation
    from src.libVis.torch import plot_keypoints_batch, plot_Kabsch

    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from src.models.ransac import RANSAC

    with initialize(config_path="../../configs/"):
        cfg = compose(config_name="train.yaml")
    OmegaConf.set_struct(cfg, False)

    save_dir = "./tmp"
    os.makedirs(save_dir, exist_ok=True)

    cfg.machine.batch_size = 12
    cfg.data.train.dataloader.batch_size = cfg.machine.batch_size
    cfg.data.train.dataloader.dataset_name = "gso"
    train_dataset = instantiate(cfg.data.train.dataloader)

    dataloader = NoneFilteringDataLoader(
        train_dataset.web_dataloader.datapipeline,
        batch_size=cfg.machine.batch_size,
        num_workers=10,
        collate_fn=train_dataset.collate_fn,
    )
    # dataloader = list(tqdm(dataloader))
    ransac = RANSAC(pixel_threshold=5)
    start_time = time.time()
    for idx, batch in enumerate(dataloader):
        batch = batch.cuda()
        end_time = time.time()
        print(f"Time: {end_time - start_time}")
        start_time = end_time

        keypoint_img = plot_keypoints_batch(batch, concate_input_in_pred=False)
        batch.relScale = batch.relScale.unsqueeze(1).repeat(1, 256)
        batch.relInplane = batch.relInplane.unsqueeze(1).repeat(1, 256)

        # vis relative scale and inplane
        M, idx_failed, _ = ransac(batch)
        wrap_img = plot_Kabsch(batch, M)
        vis_img = torch.cat([keypoint_img, wrap_img], dim=3)
        save_image(
            vis_img,
            os.path.join(save_dir, f"{idx:06d}.png"),
            nrow=4,
        )
