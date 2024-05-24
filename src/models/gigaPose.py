import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from einops import repeat
import pytorch_lightning as pl
from tqdm import tqdm
import pandas as pd
from src.utils.logging import get_logger, log_image
from src.utils.batch import BatchedData, gather
from src.utils.optimizer import HybridOptim
from torchvision.utils import save_image
from src.utils.time import Timer
from src.models.loss import cosine_similarity
from src.lib3d.torch import (
    cosSin,
    get_relative_scale_inplane,
    geodesic_distance,
)
from src.libVis.torch import (
    plot_Kabsch,
    plot_keypoints_batch,
    save_tensor_to_image,
)
from src.models.poses import ObjectPoseRecovery
import src.megapose.utils.tensor_collection as tc
from src.utils.inout import save_predictions_from_batched_predictions

logger = get_logger(__name__)


class GigaPose(pl.LightningModule):
    def __init__(
        self,
        model_name,
        ae_net,
        ist_net,
        training_loss,
        testing_metric,
        optim_config,
        log_interval,
        log_dir,
        max_num_dets_per_forward=None,
        **kwargs,
    ):
        # define the network
        super().__init__()
        self.model_name = model_name
        self.ae_net = ae_net
        self.ist_net = ist_net
        self.training_loss = training_loss
        self.testing_metric = testing_metric

        self.max_num_dets_per_forward = max_num_dets_per_forward

        self.log_interval = log_interval
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(osp.join(self.log_dir, "predictions"), exist_ok=True)

        self.optim_config = optim_config
        self.optim_name = "AdamW"

        # for testing
        self.template_datas = {}
        self.pose_recovery = {}
        self.l2_loss = nn.MSELoss()
        self.timer = Timer()
        self.run_id = None
        self.template_datasets = None
        self.test_dataset_name = None

        logger.info("Initialize GigaPose done!")

    def warm_up_lr(self):
        for optim in self.trainer.optimizers:
            for idx_group, pg in enumerate(optim.param_groups):
                if len(optim.param_groups) > 1:
                    lr = self.lr["ae"] if idx_group == 0 else self.lr["ist"]
                    pg["lr"] = (
                        self.global_step / float(self.optim_config.warm_up_steps) * lr
                    )
                else:
                    pg["lr"] = (
                        self.global_step
                        / float(self.optim_config.warm_up_steps)
                        * self.lr
                    )
            if self.global_step % 50 == 0:
                logger.info(f"Step={self.global_step}, lr warm up: lr={pg['lr']}")

    def configure_optimizers(self):
        # define optimizer
        if self.optim_config.nets_to_train in ["ae", "all"]:
            logger.info("Optimizer for ae net")
            ae_optimizer = torch.optim.AdamW(
                self.ae_net.get_toUpdate_parameters(),
                self.optim_config.ae_lr,
                weight_decay=self.optim_config.weight_decay,
            )
            self.lr = self.optim_config.ae_lr

        if self.optim_config.nets_to_train in ["ist", "all"]:
            logger.info("Optimizer for ist net")
            ist_optimizer = torch.optim.AdamW(
                self.ist_net.parameters(),
                self.optim_config.ist_lr,
                weight_decay=self.optim_config.weight_decay,
            )
            self.lr = self.optim_config.ist_lr

        # disable gradient for non-updatable parameters
        if self.optim_config.nets_to_train != "all":
            if self.optim_config.nets_to_train == "ae":
                for param in self.ist_net.parameters():
                    param.requires_grad = False
            if self.optim_config.nets_to_train == "ist":
                for param in self.ae_net.parameters():
                    param.requires_grad = False
        else:
            self.lr = {
                "ae": self.optim_config.ae_lr,
                "ist": self.optim_config.ist_lr,
            }
        # combine optimizers
        if self.optim_config.nets_to_train == "all":
            optimizer = HybridOptim([ae_optimizer, ist_optimizer])
        elif self.optim_config.nets_to_train == "ae":
            optimizer = ae_optimizer
        elif self.optim_config.nets_to_train == "ist":
            optimizer = ist_optimizer
        else:
            raise NotImplementedError
        assert optimizer is not None
        return optimizer

    def move_to_device(self):
        self.ae_net.to(self.device)
        self.ist_net.to(self.device)
        logger.info(f"Moving models to {self.device} done!")

    def compute_contrastive_loss(self, batch, split):
        """
        Contrastive loss based on corresponding patches
        - Positive are patches from the same correspondence
        - Negative are patches from different correspondences
        """
        device = batch.src_img.device
        # get the query and ref features
        src_feat = self.ae_net(batch.src_img)
        tar_feat = self.ae_net(batch.tar_img)
        src_pts = getattr(batch, "src_pts").clone().long()
        tar_pts = getattr(batch, "tar_pts").clone().long()

        loss = {}
        # select the corresponding patches
        src_feat_ = gather(src_feat, src_pts)
        tar_feat_ = gather(tar_feat, tar_pts)
        label = torch.arange(src_feat_.shape[0], dtype=torch.long).to(device)
        loss["infoNCE"] = self.training_loss.contrast_loss(
            src_feat_,
            tar_feat_,
            label,
        )

        # monitor the similarity between query and ref
        with torch.no_grad():
            pos_sim = F.cosine_similarity(src_feat_, tar_feat_, dim=1, eps=1e-8)
            loss["pos_sim"] = pos_sim.mean()

            neg_sim = cosine_similarity(src_feat_, tar_feat_, normalize=True)
            loss["neg_sim"] = neg_sim.mean()

        for metric_name, metric_value in loss.items():
            name = f"{split}/{metric_name}"
            if metric_name == "infoNCE":
                prog_bar = True
            else:
                prog_bar = False
            self.log(
                name,
                metric_value,
                sync_dist=True,
                on_step=True,
                on_epoch=False,
                prog_bar=prog_bar,
            )
        return loss

    def compute_regression_loss(self, batch, split):
        """
        Contrastive loss based on corresponding patches
        - Positive are patches from the same correspondence
        - Negative are patches from different correspondences
        """
        # get the query and ref features
        num_patches = batch.src_pts.shape[1]
        H, W = np.sqrt(num_patches).astype(int), np.sqrt(num_patches).astype(int)

        loss = {}
        gt_relInplane = getattr(batch, "relInplane")
        gt_relScale = getattr(batch, "relScale")

        src_pts = getattr(batch, "src_pts").clone().long()
        tar_pts = getattr(batch, "tar_pts").clone().long()

        preds = self.ist_net(
            src_img=batch.src_img,
            tar_img=batch.tar_img,
            src_pts=src_pts,
            tar_pts=tar_pts,
        )
        if preds["inplane"].shape[0] != src_pts.shape[0]:
            gt_relInplane = repeat(gt_relInplane, "b -> b 1 H W", H=H, W=W)
            gt_relInplane = gather(gt_relInplane, src_pts).squeeze(1)

            gt_relScale = repeat(gt_relScale, "b -> b 1 H W", H=H, W=W)
            gt_relScale = gather(gt_relScale, src_pts).squeeze(1)

        # it is simpler to use l2 loss for warm up to regress correct magnitudes
        if self.trainer.global_step < self.optim_config.warm_up_steps:
            loss["inp"] = self.l2_loss(
                preds["inplane"],
                cosSin(gt_relInplane),
            )
            loss["scale"] = self.l2_loss(preds["scale"], gt_relScale)
        else:
            loss["inp"] = self.training_loss.inplane_loss(
                preds["inplane"],
                cosSin(gt_relInplane),
            )
            loss["scale"] = self.training_loss.scale_loss(preds["scale"], gt_relScale)

        # Visualize the predicted inplane and scale
        with torch.no_grad():
            scale_err = torch.abs(preds["scale"].clone() - gt_relScale)
            loss["scale_err"] = scale_err.mean()

            angle_err = geodesic_distance(preds["inplane"], cosSin(gt_relInplane))
            loss["angle_err"] = torch.rad2deg(angle_err)

        for metric_name, metric_value in loss.items():
            name = f"{split}/{metric_name}"
            if metric_name in ["inp", "scale"]:
                prog_bar = True
            else:
                prog_bar = False
            self.log(
                name,
                metric_value,
                sync_dist=True,
                on_step=True,
                on_epoch=False,
                prog_bar=prog_bar,
            )
        return loss

    def training_step(self, batchs, idx_batch):
        if self.trainer.global_step < self.optim_config.warm_up_steps:
            self.warm_up_lr()
        elif self.trainer.global_step == self.optim_config.warm_up_steps:
            logger.info(f"Finished warm up, setting lr to {self.lr}")

        loss = 0
        times = {}

        for idx_dataset, batch in enumerate(batchs):
            if batch is None:
                continue
            if idx_batch % self.log_interval == 0:
                vis_pts = plot_keypoints_batch(batch)
                sample_path = f"{self.log_dir}/sample_rank{self.global_rank}.png"
                save_tensor_to_image(vis_pts, sample_path)
                log_image(
                    logger=self.logger,
                    name=f"vis/train_samples_{idx_dataset}",
                    path=sample_path,
                )

            if self.optim_config.nets_to_train in ["ist", "all"]:
                self.timer.tic()
                loss_ = self.compute_regression_loss(batch, "train")
                loss += loss_["scale"] + loss_["inp"]
                times[f"scale_inp_{idx_dataset}"] = self.timer.toc()

            if self.optim_config.nets_to_train in ["ae", "all"]:
                self.timer.tic()
                loss_ = self.compute_contrastive_loss(batch, "train")
                loss += loss_["infoNCE"]
                times[f"infoNCE_{idx_dataset}"] = self.timer.toc()

        for time_name, time_value in times.items():
            self.log(
                time_name,
                time_value,
                sync_dist=True,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )

        self.log(
            "total",
            loss,
            sync_dist=True,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return loss

    def validate_contrast_loss(self, batch, idx_batch, split):
        src_feat = self.ae_net(batch.src_img)
        tar_feat = self.ae_net(batch.tar_img)

        preds = self.testing_metric.val(
            src_feat=src_feat,
            tar_feat=tar_feat,
            src_mask=batch.src_mask,
            tar_mask=batch.tar_mask,
        )
        setattr(batch, "pred_src_pts", preds.src_pts)
        setattr(batch, "pred_tar_pts", preds.tar_pts)

        # monitor the distance between the gt and the predicted matches for same target patches
        mask = torch.logical_and(
            batch.tar_pts[:, :, 1] != -1, batch.pred_tar_pts[:, :, 1] != -1
        )
        distance = (batch.tar_pts[mask] - batch.pred_tar_pts[mask]).norm(dim=1).mean()
        self.log(
            f"{split}/matching",
            distance,
            sync_dist=True,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        # visualize matches
        vis_pts = plot_keypoints_batch(batch, type_data="pred")
        sample_path = f"{self.log_dir}/{split}_sample_rank{self.global_rank}.png"
        save_tensor_to_image(vis_pts, sample_path)
        log_image(
            logger=self.logger,
            name=f"vis/{split}_samples",
            path=sample_path,
        )

    def validation_step(self, batch, idx_batch):
        _ = self.compute_regression_loss(batch, "val")
        _ = self.validate_contrast_loss(batch, idx_batch, "val")

    def set_template_data(self, dataset_name):
        logger.info("Initializing template data ...")
        self.timer.tic()
        template_dataset = self.template_datasets[dataset_name]
        names = ["rgb", "mask", "K", "M", "poses", "ae_features", "ist_features"]
        template_data = {name: BatchedData(None) for name in names}

        for idx in tqdm(range(len(template_dataset))):
            for name in names:
                if name in ["ae_features", "ist_features"]:
                    continue
                if name == "rgb":
                    templates = template_dataset[idx].rgb.to(self.device)
                    if self.max_num_dets_per_forward is None:
                        template_data[name].append(templates)

                    ae_features = self.ae_net(templates)
                    template_data["ae_features"].append(ae_features)

                    ist_features = self.ist_net.forward_by_chunk(templates)
                    template_data["ist_features"].append(ist_features)
                else:
                    tmp = getattr(template_dataset[idx], name)
                    template_data[name].append(tmp.to(self.device))
        if self.max_num_dets_per_forward is not None:
            names.remove("rgb")
        for name in names:
            template_data[name].stack()
            template_data[name] = template_data[name].data

        self.template_datas[dataset_name] = tc.PandasTensorCollection(
            infos=pd.DataFrame(), **template_data
        )
        self.pose_recovery[dataset_name] = ObjectPoseRecovery(
            template_K=template_data["K"],
            template_Ms=template_data["M"],
            template_poses=template_data["poses"],
        )
        num_obj = len(template_data["K"])
        onboarding_time = self.timer.toc() / num_obj
        self.timer.reset()
        logger.info(f"Init {dataset_name} done! Avg time={onboarding_time} s/object")

    def filter_and_save(
        self,
        predictions,
        test_list,
        time,
        save_path,
        keep_only_testing_instances=True,
    ):
        labels = np.asarray(predictions.infos.label).astype(np.int32)
        assert len(np.unique(labels)) == len(np.unique(test_list.infos.obj_id))
        detection_times = []
        if keep_only_testing_instances:
            selected_idxs = []
            for idx, id in enumerate(test_list.infos.obj_id):
                num_inst = test_list.infos.inst_count[idx]
                idx_inst = labels == id
                pred_inst = predictions[idx_inst.tolist()]

                selected_idx = torch.argsort(pred_inst.scores[:, 0], descending=True)
                selected_idx = selected_idx[:num_inst].cpu().numpy()
                selected_idx = np.arange(len(labels))[idx_inst][selected_idx]
                selected_idxs.extend(selected_idx.tolist())
                detection_times.extend(
                    [test_list.infos.detection_time[idx] for _ in range(num_inst)]
                )
        predictions = predictions[selected_idxs]

        detection_times = np.array(detection_times)
        detection_times = torch.from_numpy(detection_times).to(
            predictions.scores.device
        )
        time = torch.ones_like(detection_times) * time
        predictions.register_tensor("detection_time", detection_times)
        predictions.register_tensor("time", time)

        scene_id = np.asarray(predictions.infos.scene_id).astype(np.int32)
        im_id = np.asarray(predictions.infos.view_id).astype(np.int32)
        label = np.asarray(predictions.infos.label).astype(np.int32)

        np.savez(
            save_path,
            scene_id=scene_id,
            im_id=im_id,
            object_id=label,
            time=predictions.time.cpu().numpy(),
            detection_time=predictions.detection_time.cpu().numpy(),
            poses=predictions.pred_poses.cpu().numpy(),
            scores=predictions.scores.cpu().numpy(),
        )
        return selected_idxs, predictions

    def vis_retrieval(
        self, template_data, batch, selected_idxs, predictions, idx_batch
    ):
        device = template_data.rgb.device
        idx_sample = torch.arange(0, predictions.id_src.shape[0], device=device)
        tar_label_np = np.asarray(predictions.infos.label).astype(np.int32)
        tar_label = torch.from_numpy(tar_label_np).to(device)

        src_imgs = template_data.rgb[tar_label - 1]
        src_masks = template_data.mask[tar_label - 1]
        tar_img = batch.tar_img[selected_idxs]
        tar_mask = batch.tar_mask[selected_idxs]
        pred_imgs = []
        for idx_k in range(self.testing_metric.k):
            batch = tc.PandasTensorCollection(
                infos=pd.DataFrame(),
                src_img=src_imgs[idx_sample, predictions.id_src[:, idx_k]].clone(),
                src_mask=src_masks[idx_sample, predictions.id_src[:, idx_k]].clone(),
                tar_img=tar_img,
                tar_mask=tar_mask,
                src_pts=predictions.ransac_src_pts[:, idx_k],
                tar_pts=predictions.ransac_tar_pts[:, idx_k],
            )
            keypoint_img = plot_keypoints_batch(batch, concate_input_in_pred=False)
            wrap_img = plot_Kabsch(batch, predictions.M[:, idx_k])
            pred_img = torch.cat([keypoint_img, wrap_img], dim=3)
            pred_imgs.append(pred_img)
        pred_imgs = torch.cat(pred_imgs, dim=0)
        return pred_imgs

    def eval_retrieval(
        self,
        batch,
        idx_batch,
        dataset_name,
        sort_pred_by_inliers=True,
    ):
        torch.cuda.empty_cache()
        # prepare template data
        if dataset_name not in self.template_datas:
            self.set_template_data(dataset_name)

        template_data = self.template_datas[dataset_name]
        pose_recovery = self.pose_recovery[dataset_name]
        times = {"neighbor_search": None, "final_step": None}

        B, C, H, W = batch.tar_img.shape
        device = batch.tar_img.device

        # if low_memory_mode, two detections are forward at a time
        list_idx_sample = []
        if self.max_num_dets_per_forward is not None:
            for start_idx in np.arange(0, B, self.max_num_dets_per_forward):
                end_idx = min(start_idx + self.max_num_dets_per_forward, B)
                idx_sample_ = torch.arange(start_idx, end_idx, device=device)
                list_idx_sample.append(idx_sample_)
        else:
            idx_sample = torch.arange(0, B, device=device)
            list_idx_sample.append(idx_sample)

        for idx_sub_batch, idx_sample in enumerate(list_idx_sample):
            # compute target features
            tar_ae_features = self.ae_net(batch.tar_img[idx_sample])
            tar_label_np = np.asarray(
                batch.infos.label[idx_sample.cpu().numpy()]
            ).astype(np.int32)
            tar_label = torch.from_numpy(tar_label_np).to(device)

            # template data
            src_ae_features = template_data.ae_features[tar_label - 1]
            src_masks = template_data.mask[tar_label - 1]

            # Step 1: Nearest neighbor search
            self.timer.tic()
            predictions_ = self.testing_metric.test(
                src_feats=src_ae_features,
                tar_feat=tar_ae_features,
                src_masks=src_masks,
                tar_mask=batch.tar_mask[idx_sample],
                max_batch_size=None,
            )
            predictions_.infos = batch.infos
            if idx_sub_batch == 0:
                predictions = predictions_
            else:
                predictions.cat_df(predictions_)

        # Step 2: Find affine transforms
        num_patches = predictions.src_pts.shape[2]
        k = self.testing_metric.k
        pred_scales = torch.zeros(B, k, num_patches, device=device)
        pred_cosSin_inplanes = torch.zeros(B, k, num_patches, 2, device=device)

        self.timer.tic()
        for idx_k in range(k):
            idx_sample = torch.arange(0, B, device=device)
            idx_views = [idx_sample, predictions.id_src[:, idx_k]]

            tar_label_np = np.asarray(batch.infos.label).astype(np.int32)
            tar_label = torch.from_numpy(tar_label_np).to(device)

            src_ist_features = template_data.ist_features[tar_label - 1]
            tar_ist_features = self.ist_net.forward_by_chunk(batch.tar_img[idx_sample])

            if self.max_num_dets_per_forward is not None:
                (
                    pred_scales[:, idx_k],
                    pred_cosSin_inplanes[:, idx_k],
                ) = self.ist_net.inference_by_chunk(
                    src_feat=src_ist_features[idx_views],
                    tar_feat=tar_ist_features,
                    src_pts=predictions.src_pts[:, idx_k],
                    tar_pts=predictions.tar_pts[:, idx_k],
                    max_batch_size=self.max_num_dets_per_forward,
                )
            else:
                (
                    pred_scales[:, idx_k],
                    pred_cosSin_inplanes[:, idx_k],
                ) = self.ist_net.inference(
                    src_feat=src_ist_features[idx_views],
                    tar_feat=tar_ist_features,
                    src_pts=predictions.src_pts[:, idx_k],
                    tar_pts=predictions.tar_pts[:, idx_k],
                )

        predictions.register_tensor("relScale", pred_scales)
        predictions.register_tensor(
            "relInplane",
            pred_cosSin_inplanes,
        )
        times["neighbor_search"] = self.timer.toc()
        self.timer.reset()

        self.timer.tic()
        predictions = pose_recovery.forward_ransac(predictions=predictions)
        # sort the predictions by the number of inliers for each detection
        score = torch.sum(predictions.ransac_scores, dim=2) / num_patches
        predictions.register_tensor("scores", score)
        if sort_pred_by_inliers:
            order = torch.argsort(score, dim=1, descending=True)
            for k, v in predictions._tensors.items():
                if k in ["infos", "meta"]:
                    continue
                predictions.register_tensor(k, v[idx_sample[:, None], order])

        # calculate prediction
        pred_poses = self.pose_recovery[dataset_name].forward_recovery(
            tar_label=tar_label,
            tar_K=batch.tar_K,
            tar_M=batch.tar_M,
            pred_src_views=predictions.id_src,
            pred_M=predictions.M.clone(),
        )
        predictions.register_tensor("pred_poses", pred_poses)

        times["final_step"] = self.timer.toc()
        self.timer.reset()
        total_time = sum(times.values())

        save_path = osp.join(self.log_dir, "predictions", f"{idx_batch}.npz")
        selected_idxs, predictions = self.filter_and_save(
            predictions, test_list=batch.test_list, time=total_time, save_path=save_path
        )

        if idx_batch % self.log_interval == 0 and self.max_num_dets_per_forward is None:
            vis_img = self.vis_retrieval(
                template_data=template_data,
                batch=batch,
                selected_idxs=selected_idxs,
                predictions=predictions,
                idx_batch=idx_batch,
            )
            sample_path = f"{self.log_dir}/retrieved_sample_rank{self.global_rank}_{idx_batch}.png"
            save_image(
                vis_img,
                sample_path,
                nrow=predictions.id_src.shape[0],
            )
            log_image(
                logger=self.logger,
                name=f"{dataset_name}",
                path=sample_path,
            )

    @torch.no_grad()
    def test_step(self, batch, idx_batch):
        self.eval_retrieval(
            batch,
            idx_batch=idx_batch,
            dataset_name=self.test_dataset_name,
        )
        return 0

    def on_test_epoch_end(self):
        if self.global_rank == 0:
            prediction_dir = osp.join(self.log_dir, "predictions")
            save_predictions_from_batched_predictions(
                prediction_dir,
                dataset_name=self.test_dataset_name,
                model_name=self.model_name,
                run_id=self.run_id,
                is_refined=False,
            )
