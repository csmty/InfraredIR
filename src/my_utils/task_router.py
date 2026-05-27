import cv2
import numpy as np
import torch
from collections import OrderedDict
from torchvision import transforms
from omegaconf import OmegaConf
from ultralytics import YOLO

from my_utils.training_utils import collate_fn_detection
from my_utils.task_loss import YOLOv8LossCalculator, SegFormerLossCalculator
from my_utils.yolo_utils import (
    make_yolo_targets_from_batch,
    select_targets_for_image,
    draw_gt_boxes,
    draw_pred_boxes,
)

from nets.segformer.test import colorize_mask_ts
from nets.SCTransNet.SCTransNet import SCTransNet
import nets.SCTransNet.Config as config_sc


VALID_TASK_TYPES = ("enhancement", "detection", "segmentation", "small_targets")


def get_task_type(config):
    task_type = config.get("task_type", "enhancement")
    if task_type not in VALID_TASK_TYPES:
        raise ValueError(
            f"Unsupported task_type: {task_type}. "
            f"Expected one of {VALID_TASK_TYPES}."
        )
    return task_type


def get_task_collate_fn(task_type):
    if task_type == "detection":
        return collate_fn_detection
    return None

def _require_arg(args, arg_name, task_type):
    value = getattr(args, arg_name, None)
    if value is None or str(value).strip() == "":
        raise ValueError(
            f"`{arg_name}` is required when task_type='{task_type}', "
            f"but got empty value."
        )
    return value

def build_task_components(task_type, args):
    if task_type == "enhancement":
        return _build_enhancement_components(args)
    elif task_type == "detection":
        return _build_detection_components(args)
    elif task_type == "segmentation":
        return _build_segmentation_components(args)
    elif task_type == "small_targets":
        return _build_small_targets_components(args)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

def _build_enhancement_components(args):
    return {
        "task_model": None,
        "loss_calculator": None,
    }

def _build_detection_components(args):
    yolo_path = _require_arg(args, "yolo_weight", "detection")

    net_yolo = YOLO(yolo_path)
    net_yolo.to("cuda")
    net_yolo.eval()

    class_names = net_yolo.names if hasattr(net_yolo, "names") else None
    loss_calculator = YOLOv8LossCalculator(yolo_path)

    return {
        "task_model": net_yolo,
        "loss_calculator": loss_calculator,
        "class_names": class_names,
    }


def _build_segmentation_components(args):
    segformer_config = _require_arg(args, "segformer_config", "segmentation")
    segformer_weight = _require_arg(args, "segformer_weight", "segmentation")

    seg_cfg = OmegaConf.load(segformer_config)
    loss_calculator = SegFormerLossCalculator(seg_cfg, segformer_weight)

    return {
        "task_model": None,
        "loss_calculator": loss_calculator,
    }


def _build_small_targets_components(args):
    sctransnet_weight = _require_arg(args, "sctransnet_weight", "small_targets")

    config_vit = config_sc.get_SCTrans_config()
    net_sc = SCTransNet(config_vit, mode="test", deepsuper=True)

    state_dict = torch.load(sctransnet_weight, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in state_dict["state_dict"].items():
        name = k[6:]  # remove 'module.'
        new_state_dict[name] = v

    net_sc.load_state_dict(new_state_dict)
    net_sc.eval()

    for param in net_sc.parameters():
        param.requires_grad = False

    return {
        "task_model": net_sc,
        "criterion": torch.nn.BCELoss(),
        "img_norm_cfg": {
            "mean": 101.06385040283203,
            "std": 34.619606018066406,
        },
    }


def prepare_task_training_objects(task_type, task_ctx, accelerator,
                                  net_iir, optimizer, dl_train, dl_val, lr_scheduler):
    if task_type == "small_targets" and task_ctx.get("task_model") is not None:
        net_iir, task_ctx["task_model"], optimizer, dl_train, dl_val, lr_scheduler = accelerator.prepare(
            net_iir, task_ctx["task_model"], optimizer, dl_train, dl_val, lr_scheduler
        )
    else:
        net_iir, optimizer, dl_train, dl_val, lr_scheduler = accelerator.prepare(
            net_iir, optimizer, dl_train, dl_val, lr_scheduler
        )

    return net_iir, task_ctx, optimizer, dl_train, dl_val, lr_scheduler

def _try_move_to_device(obj, device, dtype=None):
    if obj is None:
        return
    if hasattr(obj, "to"):
        try:
            if dtype is None:
                obj.to(device)
            else:
                obj.to(device, dtype=dtype)
        except TypeError:
            obj.to(device)

def move_task_components_to_device(task_type, task_ctx, device, dtype):
    if task_type == "enhancement":
        return task_ctx

    if task_type == "detection":
        _try_move_to_device(task_ctx.get("task_model"), device)
        if task_ctx.get("task_model") is not None:
            task_ctx["task_model"].eval()
        _try_move_to_device(task_ctx.get("loss_calculator"), device)

    elif task_type == "segmentation":
        _try_move_to_device(task_ctx.get("loss_calculator"), device)

    elif task_type == "small_targets":
        _try_move_to_device(task_ctx.get("task_model"), device, dtype=dtype)
        if task_ctx.get("task_model") is not None:
            task_ctx["task_model"].eval()

    return task_ctx


def compute_task_loss(task_type, task_ctx, x_tgt_pred, batch):
    if task_type == "enhancement":
        return _compute_enhancement_loss(task_ctx, x_tgt_pred, batch)
    elif task_type == "detection":
        return _compute_detection_loss(task_ctx, x_tgt_pred, batch)
    elif task_type == "segmentation":
        return _compute_segmentation_loss(task_ctx, x_tgt_pred, batch)
    elif task_type == "small_targets":
        return _compute_small_targets_loss(task_ctx, x_tgt_pred, batch)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

def _compute_enhancement_loss(task_ctx, x_tgt_pred, batch):
    return x_tgt_pred.new_zeros(()), {}

def _compute_detection_loss(task_ctx, x_tgt_pred, batch):
    targets = make_yolo_targets_from_batch(batch)
    x_input = x_tgt_pred * 0.5 + 0.5
    loss_task = task_ctx["loss_calculator"].calculate_batch_loss(x_input, targets)

    task_outputs = {
        "targets": targets
    }
    return loss_task, task_outputs


def _compute_segmentation_loss(task_ctx, x_tgt_pred, batch):
    x_mask = batch["mask"]
    loss_task, x_pred_seg = task_ctx["loss_calculator"].calculate_batch_loss(
        x_tgt_pred * 0.5 + 0.5,
        x_mask
    )

    task_outputs = {
        "pred_map": x_pred_seg
    }
    return loss_task, task_outputs


def _normalize_small_target_input(x_tgt_pred, img_norm_cfg):
    x_input = x_tgt_pred[:, :1, :, :]
    x_input = x_input * 0.5 + 0.5
    x_input = x_input * 255.0
    x_input = (x_input - img_norm_cfg["mean"]) / img_norm_cfg["std"]
    return x_input


def _compute_small_targets_loss(task_ctx, x_tgt_pred, batch):
    x_mask = batch["mask"]

    x_input = _normalize_small_target_input(x_tgt_pred, task_ctx["img_norm_cfg"])

    task_ctx["task_model"].eval()
    x_pred = task_ctx["task_model"](x_input)

    loss_task = task_ctx["criterion"](x_pred, x_mask)

    task_outputs = {
        "pred_map": x_pred
    }
    return loss_task, task_outputs


def save_task_visualization(task_type, task_ctx, batch,
                            x_src, x_tgt, x_tgt_pred, task_outputs, save_path):
    if task_type == "enhancement":
        _save_enhancement_visualization(batch, x_src, x_tgt, x_tgt_pred, save_path)
    elif task_type == "detection":
        _save_detection_visualization(task_ctx, batch, x_src, x_tgt, x_tgt_pred, save_path)
    elif task_type == "segmentation":
        _save_segmentation_visualization(batch, x_src, x_tgt, x_tgt_pred, task_outputs, save_path)
    elif task_type == "small_targets":
        _save_small_targets_visualization(batch, x_src, x_tgt, x_tgt_pred, task_outputs, save_path)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")


def _save_enhancement_visualization(batch, x_src, x_tgt, x_tgt_pred, save_path):
    x_src_vis = x_src[:, 0:1, :, :].cpu().detach() * 0.5 + 0.5
    x_tgt_vis = x_tgt[:, 0:1, :, :].cpu().detach() * 0.5 + 0.5
    x_tgt_pred_vis = x_tgt_pred[:, 0:1, :, :].cpu().detach() * 0.5 + 0.5

    combined = torch.cat(
        [x_src_vis, x_tgt_vis, x_tgt_pred_vis],
        dim=3,
    )
    output_pil = transforms.ToPILImage()(combined[0])
    output_pil.save(save_path)


def _save_detection_visualization(task_ctx, batch, x_src, x_tgt, x_tgt_pred, save_path):
    x_src_vis = x_src.cpu().detach() * 0.5 + 0.5
    x_tgt_vis = x_tgt.cpu().detach() * 0.5 + 0.5
    x_tgt_pred_vis = x_tgt_pred.cpu().detach() * 0.5 + 0.5

    with torch.no_grad():
        x_input = x_tgt_pred * 0.5 + 0.5
        results = task_ctx["task_model"].predict(
            x_input,
            imgsz=x_input.shape[2],
            verbose=False
        )

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            pred_boxes = results[0].boxes.xyxy.detach().cpu().numpy()
            pred_scores = results[0].boxes.conf.detach().cpu().numpy()
            pred_classes = results[0].boxes.cls.detach().cpu().numpy()
        else:
            pred_boxes = np.zeros((0, 4), dtype=np.float32)
            pred_scores = np.zeros((0,), dtype=np.float32)
            pred_classes = np.zeros((0,), dtype=np.float32)

    targets = make_yolo_targets_from_batch(batch)

    cls_vis, bboxes_vis = select_targets_for_image(
        batch_idx=targets["batch_idx"].cpu(),
        cls=targets["cls"].cpu(),
        bboxes=targets["bboxes"].cpu(),
        image_idx=0,
    )

    x_tgt_box_vis = draw_gt_boxes(
        x_tgt_vis,
        cls_vis,
        bboxes_vis,
        class_names=task_ctx["class_names"],
    )

    x_tgt_pred_box_vis = draw_pred_boxes(
        x_tgt_pred_vis,
        pred_boxes,
        pred_scores,
        pred_classes,
        class_names=task_ctx["class_names"],
        score_thr=0.25,
    )

    combined = torch.cat(
        [x_src_vis, x_tgt_vis, x_tgt_pred_vis, x_tgt_box_vis, x_tgt_pred_box_vis],
        dim=3
    )
    output_pil = transforms.ToPILImage()(combined[0])
    output_pil.save(save_path)


def _save_segmentation_visualization(batch, x_src, x_tgt, x_tgt_pred, task_outputs, save_path):
    x_src_vis = x_src.cpu().detach() * 0.5 + 0.5
    x_tgt_vis = x_tgt.cpu().detach() * 0.5 + 0.5
    x_tgt_pred_vis = x_tgt_pred.cpu().detach() * 0.5 + 0.5

    x_mask_color = colorize_mask_ts(batch["mask"]).cpu().detach()
    x_pred_seg_color = colorize_mask_ts(task_outputs["pred_map"]).cpu().detach()

    combined = torch.cat(
        [x_src_vis, x_tgt_vis, x_tgt_pred_vis, x_mask_color, x_pred_seg_color],
        dim=3
    )
    output_pil = transforms.ToPILImage()(combined[0])
    output_pil.save(save_path)


def _save_small_targets_visualization(batch, x_src, x_tgt, x_tgt_pred, task_outputs, save_path):
    x_src_vis = x_src[:, 0:1, :, :].cpu().detach() * 0.5 + 0.5
    x_tgt_vis = x_tgt[:, 0:1, :, :].cpu().detach() * 0.5 + 0.5
    x_tgt_pred_vis = x_tgt_pred[:, 0:1, :, :].cpu().detach() * 0.5 + 0.5

    x_pred_vis = task_outputs["pred_map"].cpu().detach().clamp(0, 1)
    x_mask_vis = batch["mask"].cpu().detach().clamp(0, 1)

    combined = torch.cat(
        [x_src_vis, x_tgt_vis, x_tgt_pred_vis, x_pred_vis, x_mask_vis],
        dim=3
    )

    combined_img = combined[0, 0].numpy()
    combined_img = (combined_img * 255.0).astype(np.uint8)
    cv2.imwrite(save_path, combined_img)