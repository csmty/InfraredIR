import os
import argparse

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from PIL import Image
from torch.serialization import add_safe_globals
from tqdm import tqdm

from nets.segformer.core.model import WeTr
from nets.segformer.datasets import imutils
from nets.segformer.utils.eval_seg import scores

add_safe_globals([ListConfig])

def setup_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./nets/segformer/configs/fmb.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--input_dir', type=str, required=True, help='输入图像目录')
    parser.add_argument('--gt_dir', type=str, required=True, help='GT语义分割标签目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出结果目录')
    return parser.parse_args()


def load_model(cfg, checkpoint_path):
    model = WeTr(
        backbone=cfg.exp.backbone,
        num_classes=cfg.dataset.num_classes,
        embedding_dim=256,
        pretrained=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    return model


def process_image(image_path, model, device):
    image = np.asarray(imageio.imread(image_path))
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)

    h, w, _ = image.shape
    image = imutils.img_resize_short(image, min_size=512)
    image = imutils.normalize_img(image)
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model(image)

        output = torch.nn.functional.interpolate(
            output,
            size=(h, w),
            mode='bilinear',
            align_corners=True,
        )

        output = torch.nn.functional.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1)[0].cpu().numpy()

    return pred, (h, w)


# 颜色字典（类别 -> RGB）
color_map = {
    0: (0, 0, 0),
    1: (179, 228, 228),  # road
    2: (181, 57, 133),   # sidewalk
    3: (67, 162, 177),   # building
    4: (200, 178, 50),   # lamp
    5: (132, 45, 199),   # sign
    6: (66, 172, 84),    # vegetation
    7: (179, 73, 79),    # sky
    8: (76, 99, 166),    # person
    9: (66, 121, 253),   # car
    10: (137, 165, 91),  # truck
    11: (155, 97, 152),  # bus
    12: (105, 153, 140), # motorcycle
    13: (222, 215, 158), # bicycle
    14: (135, 113, 90),  # pole
}

class_names = {
    0: 'background',
    1: 'road',
    2: 'sidewalk',
    3: 'building',
    4: 'lamp',
    5: 'sign',
    6: 'vegetation',
    7: 'sky',
    8: 'person',
    9: 'car',
    10: 'truck',
    11: 'bus',
    12: 'motorcycle',
    13: 'bicycle',
    14: 'pole',
}


def colorize_mask_np(mask: np.ndarray) -> np.ndarray:
    """将类别掩码转为彩色图像 (H, W, 3)"""
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in color_map.items():
        color_img[mask == cls_id] = color
    return color_img


def colorize_mask_ts(mask: torch.Tensor) -> torch.Tensor:
    """
    将类别掩码张量转为彩色图像张量。

    参数:
        mask: torch.Tensor, 形状为 (B, 1, H, W)，类别ID为整数

    返回:
        color_img: torch.Tensor, 形状为 (B, 3, H, W)，取值范围 [0, 255]，dtype=torch.uint8
    """
    assert mask.ndim == 4 and mask.size(1) == 1, f"mask形状应为(B,1,H,W)，但得到{mask.shape}"
    device = mask.device
    B, _, H, W = mask.shape

    num_classes = max(color_map.keys()) + 1
    lut = torch.zeros((num_classes, 3), dtype=torch.uint8, device=device)
    for k, v in color_map.items():
        lut[k] = torch.tensor(v, dtype=torch.uint8, device=device)

    mask_flat = mask.squeeze(1).long()
    color_img = lut[mask_flat]
    color_img = color_img.permute(0, 3, 1, 2)

    return color_img


def get_gt_path(img_path, input_dir, gt_dir):
    rel_path = os.path.relpath(img_path, input_dir)
    rel_dir = os.path.dirname(rel_path)
    stem = os.path.splitext(os.path.basename(rel_path))[0]
    return os.path.join(gt_dir, rel_dir, stem + '.png')


def read_gt_mask(gt_path, target_size=None):
    gt = np.asarray(Image.open(gt_path))

    if gt.ndim == 3:
        if gt.shape[2] == 1:
            gt = gt[:, :, 0]
        elif np.all(gt[:, :, 0] == gt[:, :, 1]) and np.all(gt[:, :, 0] == gt[:, :, 2]):
            gt = gt[:, :, 0]
        else:
            raise ValueError(
                f"GT标签 {gt_path} 是RGB彩色图, 不是类别ID图。"
                f"请先将彩色标注转换为单通道类别ID标签后再计算指标。"
            )

    gt = gt.astype(np.int64)

    if target_size is not None and gt.shape[:2] != target_size:
        gt_img = Image.fromarray(gt.astype(np.uint8))
        gt_img = gt_img.resize((target_size[1], target_size[0]), resample=Image.NEAREST)
        gt = np.asarray(gt_img).astype(np.int64)

    return gt


def write_metrics_txt(metric_result, metric_txt_path, num_classes, total_eval_images, missing_gts):
    os.makedirs(os.path.dirname(metric_txt_path), exist_ok=True)

    with open(metric_txt_path, 'w', encoding='utf-8') as f:
        f.write('Semantic Segmentation Evaluation Results\n')
        f.write('=' * 50 + '\n')
        f.write(f'Total evaluated images: {total_eval_images}\n')
        f.write(f'Num classes: {num_classes}\n\n')

        f.write(f"Pixel Accuracy: {metric_result['Pixel Accuracy']:.6f}\n")
        f.write(f"Mean Accuracy:  {metric_result['Mean Accuracy']:.6f}\n")
        f.write(f"Mean IoU:       {metric_result['Mean IoU']:.6f}\n\n")

        f.write('Class IoU:\n')
        for cls_id in range(num_classes):
            iou = metric_result['Class IoU'].get(cls_id, np.nan)
            cls_name = class_names.get(cls_id, f'class_{cls_id}')
            if np.isnan(iou):
                f.write(f'  {cls_id:02d} {cls_name}: nan\n')
            else:
                f.write(f'  {cls_id:02d} {cls_name}: {iou:.6f}\n')

        if missing_gts:
            f.write('\nMissing GT files, skipped for metric calculation:\n')
            for img_path, gt_path in missing_gts:
                f.write(f'  image: {img_path}\n')
                f.write(f'  gt:    {gt_path}\n')


def main():
    args = setup_arg_parser()

    cfg = OmegaConf.load(args.config)
    num_classes = int(cfg.dataset.num_classes)

    os.makedirs(args.output_dir, exist_ok=True)

    # 指标固定保存到 output_dir/miou.txt
    metric_txt_path = os.path.join(args.output_dir, 'miou.txt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(cfg, args.checkpoint)
    model = model.to(device)
    model.eval()

    print(f"设备: {device}")
    print(f"模型已加载: {args.checkpoint}")
    print(f"GT标签目录: {args.gt_dir}")
    print(f"指标将保存到: {metric_txt_path}")

    label_trues = []
    label_preds = []
    missing_gts = []

    for root, dirs, files in os.walk(args.input_dir):
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) == 0:
            continue

        rel_path = os.path.relpath(root, args.input_dir)
        output_folder = os.path.join(args.output_dir, rel_path)
        os.makedirs(output_folder, exist_ok=True)

        print(f"处理文件夹: {root}")

        for img_file in tqdm(image_files, desc=f"处理 {rel_path}"):
            img_path = os.path.join(root, img_file)

            pred, orig_size = process_image(img_path, model, device)

            # ---------- 创建子目录 ----------
            pred_dir = os.path.join(output_folder, 'pred')
            color_dir = os.path.join(output_folder, 'color')
            os.makedirs(pred_dir, exist_ok=True)
            os.makedirs(color_dir, exist_ok=True)

            # ---------- 保存原始预测（类别ID） ----------
            pred_mask = pred.astype(np.uint8)
            pred_img = Image.fromarray(pred_mask, mode='L')
            pred_path = os.path.join(pred_dir, f"{os.path.splitext(img_file)[0]}.png")
            pred_img.save(pred_path)

            # ---------- 保存上色预测 ----------
            color_pred = colorize_mask_np(pred)
            color_img = Image.fromarray(color_pred.astype(np.uint8))
            color_path = os.path.join(color_dir, f"{os.path.splitext(img_file)[0]}.png")
            color_img.save(color_path)

            # ---------- 读取GT并累计指标 ----------
            gt_path = get_gt_path(
                img_path=img_path,
                input_dir=args.input_dir,
                gt_dir=args.gt_dir,
            )

            if not os.path.exists(gt_path):
                missing_gts.append((img_path, gt_path))
                continue

            gt = read_gt_mask(gt_path, target_size=pred.shape)
            label_trues.append(gt)
            label_preds.append(pred.astype(np.int64))

    # ---------- 计算并保存指标 ----------
    if len(label_trues) == 0:
        print('没有找到任何可用于计算指标的GT标签, 请检查 --gt_dir 是否正确')
    else:
        metric_result = scores(label_trues, label_preds, num_classes=num_classes)
        write_metrics_txt(
            metric_result=metric_result,
            metric_txt_path=metric_txt_path,
            num_classes=num_classes,
            total_eval_images=len(label_trues),
            missing_gts=missing_gts,
        )
        print(f"指标计算完成，已保存到: {metric_txt_path}")
        print(f"Pixel Accuracy: {metric_result['Pixel Accuracy']:.6f}")
        print(f"Mean Accuracy:  {metric_result['Mean Accuracy']:.6f}")
        print(f"Mean IoU:       {metric_result['Mean IoU']:.6f}")

    if missing_gts:
        print(f"警告：有 {len(missing_gts)} 张图像没有找到对应GT, 已跳过这些图像的指标计算。")

    print(f"完成，结果在: {args.output_dir}")

if __name__ == '__main__':
    main()
