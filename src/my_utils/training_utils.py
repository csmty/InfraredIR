import argparse
import json
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import glob

import re
import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from pathlib import Path
from torch.utils import data as data
import torchvision.transforms.functional as TF

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop, triplet_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, random_add_speckle_noise_pt, random_add_saltpepper_noise_pt, bivariate_Gaussian

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, img2tensor_ir
from basicsr.utils.registry import DATASET_REGISTRY

from src.my_utils.degradation import deg_simple, deg_single

from src.my_utils.yolo_utils import build_detection_transform, sync_detection_aug_and_crop, _ensure_2d_tensor

def parse_args_paired_training(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()
    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_lpips", default=5.0, type=float)
    parser.add_argument("--lambda_l2", default=2.0, type=float)

    parser.add_argument("--base_config", required=True, type=str)
    parser.add_argument("--segformer_config", type=str, default=None)
    parser.add_argument('--segformer_weight', type=str, default=None)
    parser.add_argument('--yolo_weight', type=str, default=None)
    parser.add_argument('--sctransnet_weight', type=str, default=None)

    # validation eval args
    parser.add_argument("--eval_freq", default=500, type=int)
    parser.add_argument("--save_val", default=True, action="store_true")
    parser.add_argument("--num_samples_eval", type=int, default=50, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")

    # details about the model architecture
    parser.add_argument("--sd_path")
    parser.add_argument("--pretrained_path", type=str, default=None,)
    parser.add_argument("--de_net_path")
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    # parser.add_argument("--lora_rank_unet", default=32, type=int)
    # parser.add_argument("--lora_rank_vae", default=16, type=int)
    parser.add_argument("--neg_prob", default=0.05, type=float)
    parser.add_argument("--pos_prompt", type=str, default="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.")
    parser.add_argument("--neg_prompt", type=str, default="oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth")

    # training details
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=5000)
    parser.add_argument("--max_train_steps", type=int, default=50000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "piecewise_constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=0.1, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def collect_image_paths(paths_or_str):
    image_types = ('jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG', 'ppm', 'PPM', 'bmp', 'BMP', 'tif')

    if isinstance(image_types, str):
        image_types = [ext.strip().lower() for ext in image_types.split(',') if ext.strip()]
    else:
        image_types = [ext.lower() for ext in image_types]

    def _collect(p):
        return [f for f in p.rglob('*') 
                if f.is_file() and f.suffix[1:].lower() in image_types]

    if isinstance(paths_or_str, (str, Path)):
        p = Path(paths_or_str)
        return sorted(_collect(p))
    else:
        result = []
        for p in paths_or_str:
            result.extend(_collect(Path(p)))
        return sorted(result)

def collect_txt_paths(paths_or_str):
    txt_exts = ('txt',)

    def _collect(p):
        return [f for f in p.rglob('*')
                if f.is_file() and f.suffix[1:].lower() in txt_exts]
    
    if isinstance(paths_or_str, (str, Path)):
        p = Path(paths_or_str)
        return sorted(_collect(p))
    else:
        result = []
        for p in paths_or_str:
            result.extend(_collect(Path(p)))
        return sorted(result)    


def get_image_pairs(opt):
    """
    Return matched samples according to opt['task_type'].

    Supported outputs:
        enhancement:
            [(lq, gt), ...]
        segmentation / detection / small_targets:
            [(lq, gt, label_or_mask), ...]

    Expected folder keys:
        enhancement:
            dataroot_lq, dataroot_gt
        segmentation:
            dataroot_lq, dataroot_gt, dataroot_label
        detection:
            dataroot_lq, dataroot_gt, dataroot_label
        small_targets:
            dataroot_lq, dataroot_gt, dataroot_label
    """

    task = opt['task_type']
    img_ext = opt.get('image_type', 'png')
    recursive = opt.get('recursive', False)

    # def _scan_folder(root):
    #     pattern = "**/*." + img_ext if recursive else f"*.{img_ext}"
    #     return sorted(glob.glob(os.path.join(root, pattern), recursive=recursive))

    # def _scan_folder(root):
    #     root = Path(root)
    #     if recursive:
    #         return sorted(str(p) for p in root.rglob(f"*.{img_ext}"))
    #     else:
    #         return sorted(str(p) for p in root.glob(f"*.{img_ext}"))


    # -------------------- enhancement -------------------- #
    if task == "enhancement":
        root_lq = opt.get('dataroot_lq', None)
        root_gt = opt.get('dataroot_gt', None)
        if root_gt is None:
            raise ValueError("For task_type='enhancement', 'dataroot_gt' must be provided.")


        if root_lq is None:
            print("[get_image_pairs] ⚠️ Warning: 'dataroot_lq' is not provided. Only GT images will be used and LQ images will be generated on the fly.")
            gt_list = collect_image_paths(root_gt)
            return [(None, gt) for gt in gt_list]

        gt_list = collect_image_paths(root_gt)
        lq_list = collect_image_paths(root_lq)

        # Match LQ and GT by filename
        gt_dict = {os.path.basename(p): p for p in gt_list}
        pairs = []
        for lq_path in lq_list:
            name = os.path.basename(lq_path)
            if name in gt_dict:
                pairs.append((lq_path, gt_dict[name]))
        if not pairs:
            raise ValueError("No matched LQ-GT pairs were found. Please check the file names and paths.")
        return pairs

    # -------------------- segmentation -------------------- #
    elif task == "segmentation":
        root_lq   = opt.get('dataroot_lq', None)
        root_gt   = opt.get('dataroot_gt', None)
        root_mask = opt.get('dataroot_label', None)
        if root_gt is None or root_mask is None:
            raise ValueError("For task_type='segmentation', both 'dataroot_gt' and 'dataroot_label' must be provided.")

        lq_list   = collect_image_paths(root_lq) if root_lq else []
        gt_list   = collect_image_paths(root_gt)
        mask_list = collect_image_paths(root_mask)

        gt_dict   = {os.path.basename(p): p for p in gt_list}
        mask_dict = {os.path.basename(p): p for p in mask_list}
        pairs = []

        # Match GT and mask by filename; use GT as LQ if no LQ is found.
        for name, gt_path in gt_dict.items():
            if name not in mask_dict:
                continue
            lq_path = None
            if lq_list:
                cand = [p for p in lq_list if os.path.basename(p) == name]
                if cand:
                    lq_path = cand[0]
            if lq_path is None:
                lq_path = gt_path
            pairs.append((lq_path, gt_path, mask_dict[name]))
        if not pairs:
            raise ValueError("No matched GT-mask pairs were found. Please check the file names and paths.")
        return pairs
    
    elif task == "detection":
        root_lq   = opt.get('dataroot_lq', None)
        root_gt   = opt.get('dataroot_gt', None)
        root_label = opt.get('dataroot_label', None)
        if root_gt is None or root_label is None:
            raise ValueError("For task_type='detection', both 'dataroot_gt' and 'dataroot_label' must be provided.")

        lq_list   = collect_image_paths(root_lq) if root_lq else []
        gt_list   = collect_image_paths(root_gt)
        label_list = collect_txt_paths(root_label)

        gt_dict   = {os.path.basename(p): p for p in gt_list}
        label_dict = {os.path.basename(p): p for p in label_list}
        pairs = []

        # Match image and detection label (.txt) by filename stem.
        for name, gt_path in gt_dict.items():
            label_name = re.sub(r'\.[^.]*$', '.txt', name)  # .* -> .txt
            if label_name not in label_dict:
                continue
            lq_path = None
            if lq_list:
                cand = [p for p in lq_list if os.path.basename(p) == name]
                if cand:
                    lq_path = cand[0]
            if lq_path is None:
                lq_path = gt_path

            pairs.append((lq_path, gt_path, label_dict[label_name]))
        if not pairs:
            raise ValueError("No matched GT-label pairs were found. Please check the file names and paths.")
        return pairs
    
    # -------------------- small_targets -------------------- #
    elif task == "small_targets":
        root_lq   = opt.get('dataroot_lq', None)
        root_gt   = opt.get('dataroot_gt', None)
        root_mask = opt.get('dataroot_label', None)
        if root_gt is None or root_mask is None:
            raise ValueError("For task_type='small_targets', both 'dataroot_gt' and 'dataroot_label' must be provided.")

        lq_list   = collect_image_paths(root_lq) if root_lq else []
        gt_list   = collect_image_paths(root_gt)
        mask_list = collect_image_paths(root_mask)

        gt_dict   = {os.path.basename(p): p for p in gt_list}
        mask_dict = {os.path.basename(p): p for p in mask_list}
        pairs = []
        for name, gt_path in gt_dict.items():
            if name not in mask_dict:
                continue
            lq_path = None
            if lq_list:
                cand = [p for p in lq_list if os.path.basename(p) == name]
                if cand:
                    lq_path = cand[0]
            if lq_path is None:
                lq_path = gt_path
            pairs.append((lq_path, gt_path, mask_dict[name]))
        if not pairs:
            raise ValueError("No matched GT-mask pairs were found. Please check the file names and paths.")
        return pairs


    else:
        raise ValueError(
            f"Unsupported task_type: {task}. "
            f"Expected one of ['enhancement', 'segmentation', 'detection', 'small_targets']."
        )
    

class PairedDataset(data.Dataset):
    """Load paired samples for enhancement, segmentation, detection, and small-target tasks."""
    def __init__(self, opt):
        super(PairedDataset, self).__init__()
        self.opt = opt
        self.task_type = opt['task_type']     # enhancement / segmentation / detection / small_targets
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.crop_size = opt.get('crop_size', 512)
        self.patch_size = (self.crop_size, self.crop_size)

        # Support both the correct key and a possible misspelling.
        self.degradation_synthetic = bool(
            opt.get('degradation_synthetic', opt.get('degradation_sysnthetic', False))
        )

        if 'image_type' not in opt:
            opt['image_type'] = 'png'

        # Unified pair format:
        # enhancement: [(lq_path, gt_path), ...]
        # segmentation/detection/small_targets: [(lq_path, gt_path, label_path), ...]
        self.pairs = get_image_pairs(opt)

        # Data augmentation settings
        self.use_hflip = self.opt.get('use_hflip', False)
        self.use_rot   = self.opt.get('use_rot', False)
        deg_file_path = "params_realesrgan.yml"

        # Detection-specific augmentation settings
        self.det_min_box_size = opt.get('det_min_box_size', 2)
        self.det_min_visibility = opt.get('det_min_visibility', 0.1)
        self.det_transform = build_detection_transform(
            crop_size=self.crop_size,
            use_hflip=self.use_hflip,
            use_rot=self.use_rot,
            min_visibility=self.det_min_visibility
        )

    # ---------------------------- Common utilities ---------------------------- #
    @staticmethod
    def _ensure_same_size(ref, x):
        if x is None:
            return None
        if ref.shape != x.shape:
            x = cv2.resize(x, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LINEAR)
        return x

    def _sync_aug_and_crop(self, imgs):
        """Apply synchronized augmentation and random crop to a list of images."""
        imgs = augment(imgs, self.use_hflip, self.use_rot)
        cs = self.crop_size
        outs = []
        for img in imgs:
            if img is None:
                outs.append(None); continue
            h, w = img.shape[:2]
            ph, pw = max(0, cs - h), max(0, cs - w)
            if ph > 0 or pw > 0:
                img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
            outs.append(img)
        # Crop all valid inputs using the same random window.
        base = next(im for im in outs if im is not None)
        H, W = base.shape[:2]
        top  = 0 if H == cs else random.randint(0, H - cs)
        left = 0 if W == cs else random.randint(0, W - cs)
        return [None if im is None else im[top:top+cs, left:left+cs, ...] for im in outs]

    def _to_tensors(self, img_lq, img_gt, img_mask=None, task=None):
        def _img_to_tensor(img):
            x = np.array(img, dtype=np.float32) / 255.0
            x = torch.from_numpy(x).unsqueeze(0).float()  # 1 x H x W
            x = (x - 0.5) / 0.5                           # [-1, 1]
            return x.repeat(3, 1, 1)                      # 3 x H x W
        lq_t = _img_to_tensor(img_lq)
        gt_t = _img_to_tensor(img_gt)
        if img_mask is None:
            return lq_t, gt_t, None
        if task == "segmentation":
            mask = (torch.from_numpy(np.array(img_mask, dtype=np.float32)).unsqueeze(0).float())
        else:
            mask = (torch.from_numpy(np.array(img_mask, dtype=np.float32)).unsqueeze(0).float()) / 255.0
        return lq_t, gt_t, mask

    def _apply_degradation(self, img_uint8):
        """Apply synthetic degradation to generate an LQ image from GT."""
        h, w = img_uint8.shape[:2]
        img = img_uint8
        
        # Current setting uses the multi-degradation pipeline.
        transform = deg_simple(p=1.0)
        return transform(image=img)["image"]

    def _read_gray(self, path):
        path = os.fspath(path)  # Path -> str
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise ValueError(f"Failed to load image: {path}")
        return im
    
    def _read_rgb(self, path):
        im = cv2.imread(path, cv2.IMREAD_COLOR)  # 读取为BGR格式
        if im is None:
            raise ValueError(f"Failed to load image: {path}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
        return im

    def _read_yolo_label(self, path):
        targets = []
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            targets.append([class_id, x_center, y_center, width, height])
        
        return np.array(targets) if targets else np.zeros((0, 5))

    # ---------------------------- Unified path loading ---------------------------- #
    def _load_paths(self, index):
        """
        Load one sample according to task type.

        Returns:
            img_lq, img_gt, img_aux, paths_dict
        where img_aux is mask or detection label if needed.
        """
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type', 'disk'), **self.io_backend_opt)

        max_attempts, attempts = len(self.pairs), 0
        while attempts < max_attempts:
            try:
                if self.task_type == "segmentation":
                    lq_path, gt_path, mask_path = self.pairs[index]
                    img_gt  = self._read_gray(gt_path)
                    img_lq  = self._read_gray(lq_path)
                    img_mask = self._read_gray(mask_path)
                    img_lq  = self._ensure_same_size(img_gt, img_lq)
                    img_mask = self._ensure_same_size(img_gt, img_mask)
                    return img_lq, img_gt, img_mask, {
                        'lq_path': lq_path, 'gt_path': gt_path, 'mask_path': mask_path
                    }
                elif self.task_type == "detection":
                    lq_path, gt_path, label_path = self.pairs[index]
                    img_gt  = self._read_gray(gt_path)
                    img_lq  = self._read_gray(lq_path)
                    img_label = self._read_yolo_label(label_path)
                    img_lq  = self._ensure_same_size(img_gt, img_lq)
                    return img_lq, img_gt, img_label, {
                        'lq_path': lq_path, 'gt_path': gt_path, 'label_path': label_path
                    }
                elif self.task_type == "small_targets":
                    lq_path, gt_path, mask_path = self.pairs[index]
                    img_gt  = self._read_gray(gt_path)
                    img_lq  = self._read_gray(lq_path)
                    img_mask = self._read_gray(mask_path)
                    img_lq  = self._ensure_same_size(img_gt, img_lq)
                    img_mask = self._ensure_same_size(img_gt, img_mask)
                    return img_lq, img_gt, img_mask, {
                        'lq_path': lq_path, 'gt_path': gt_path, 'mask_path': mask_path
                    }
                else:  # enhancement
                    lq_path, gt_path = self.pairs[index]
                    img_gt  = self._read_gray(gt_path)
                    if lq_path is None:
                        img_lq = None   # LQ will be generated from GT on the fly
                    else:
                        img_lq = self._read_gray(lq_path)
                        img_lq  = self._ensure_same_size(img_gt, img_lq)
                    return img_lq, img_gt, None, {
                        'lq_path': lq_path, 'gt_path': gt_path
                    }
            except Exception as e:
                print(f"[load] Failed to load sample {index}: {e}. Trying the next one.")
                index = (index + 1) % len(self.pairs)
                attempts += 1
        raise ValueError("All samples failed to load. Please check the dataset.")

    # ---------------------------- enhancement branch ---------------------------- #
    def _getitem_enhancement(self, index):
        img_lq_raw, img_gt_raw, _, paths = self._load_paths(index)

        if self.degradation_synthetic:
            # Augment/crop GT first, then synthesize LQ from the processed GT.
            (img_gt_aug,) = self._sync_aug_and_crop([img_gt_raw])
            img_lq_aug = self._apply_degradation(img_gt_aug.copy())
        else:
            # Apply synchronized augmentation to both LQ and GT.
            img_lq_aug, img_gt_aug = self._sync_aug_and_crop([img_lq_raw, img_gt_raw])

        lq_t, gt_t, _ = self._to_tensors(img_lq_aug, img_gt_aug, None)
        return {'lq': lq_t, 'gt': gt_t}

    # ---------------------------- segmentation branch ---------------------------- #
    def _getitem_segmentation(self, index):
        img_lq_raw, img_gt_raw, img_mask_raw, paths = self._load_paths(index)

        if self.degradation_synthetic:
            # Keep GT and mask aligned, then generate LQ from GT.
            img_gt_aug, img_mask_aug = self._sync_aug_and_crop([img_gt_raw, img_mask_raw])
            img_lq_aug = self._apply_degradation(img_gt_aug.copy())
        else:
            # Apply synchronized augmentation to LQ, GT, and mask.
            img_lq_aug, img_gt_aug, img_mask_aug = self._sync_aug_and_crop(
                [img_lq_raw, img_gt_raw, img_mask_raw]
            )

        lq_t, gt_t, mask_t = self._to_tensors(img_lq_aug, img_gt_aug, img_mask_aug, task = "segmentation")
        return {'lq': lq_t, 'gt': gt_t, 'mask': mask_t}
    
    # ---------------------------- detection branch ---------------------------- #
    def _getitem_detection(self, index):
        img_lq_raw, img_gt_raw, img_label_raw, paths = self._load_paths(index)

        if self.degradation_synthetic:
            # Apply detection-aware augmentation on GT and labels, then synthesize LQ.
            img_gt_aug, _, img_label_aug = sync_detection_aug_and_crop(
                img_gt=img_gt_raw,
                img_lq=img_gt_raw,
                label_raw=img_label_raw,
                transform=self.det_transform,
                min_box_size=self.det_min_box_size
            )
            img_lq_aug = self._apply_degradation(img_gt_aug.copy())
        else:
            # Apply synchronized detection augmentation to GT, LQ, and labels.
            img_gt_aug, img_lq_aug, img_label_aug = sync_detection_aug_and_crop(
                img_gt=img_gt_raw,
                img_lq=img_lq_raw,
                label_raw=img_label_raw,
                transform=self.det_transform,
                min_box_size=self.det_min_box_size
            )

        num_boxes = len(img_label_aug)

        cls = (
            torch.from_numpy(img_label_aug[:, 0:1].astype(np.float32))
            if num_boxes > 0 else torch.zeros((0, 1), dtype=torch.float32)
        )

        bboxes_tensor = (
            torch.from_numpy(img_label_aug[:, 1:5].astype(np.float32))
            if num_boxes > 0 else torch.zeros((0, 4), dtype=torch.float32)
        )

        lq_t, gt_t, _ = self._to_tensors(img_lq_aug, img_gt_aug)

        return {
            'lq': lq_t,
            'gt': gt_t,
            'cls': cls,
            'bboxes': bboxes_tensor,  # YOLO-format normalized xywh
        }
    # ---------------------------- small targets branch ---------------------------- #
    def _getitem_small_targets(self, index):
        img_lq_raw, img_gt_raw, img_mask_raw, paths = self._load_paths(index)

        if self.degradation_synthetic:
            img_gt_aug, img_mask_aug = self._sync_aug_and_crop([img_gt_raw, img_mask_raw])
            img_lq_aug = self._apply_degradation(img_gt_aug.copy())
        else:
            img_lq_aug, img_gt_aug, img_mask_aug = self._sync_aug_and_crop(
                [img_lq_raw, img_gt_raw, img_mask_raw]
            )

        lq_t, gt_t, mask_t = self._to_tensors(img_lq_aug, img_gt_aug, img_mask_aug)
        return {'lq': lq_t, 'gt': gt_t, 'mask': mask_t}

    # ---------------------------- Dispatch entry ---------------------------- #
    def __getitem__(self, index):
        if self.task_type == "enhancement":
            return self._getitem_enhancement(index)
        elif self.task_type == "segmentation":
            return self._getitem_segmentation(index)
        elif self.task_type == "detection":
            return self._getitem_detection(index)
        elif self.task_type == "small_targets":
            return self._getitem_small_targets(index)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def __len__(self):
        return len(self.pairs)


def randn_cropinput(lq, gt, base_size=[64, 128, 256, 512]):
    cur_size_h = random.choice(base_size)
    cur_size_w = random.choice(base_size)
    init_h = lq.size(-2)//2
    init_w = lq.size(-1)//2
    lq = lq[:, :, init_h-cur_size_h//2:init_h+cur_size_h//2, init_w-cur_size_w//2:init_w+cur_size_w//2]
    gt = gt[:, :, init_h-cur_size_h//2:init_h+cur_size_h//2, init_w-cur_size_w//2:init_w+cur_size_w//2]
    assert lq.size(-1)>=64
    assert lq.size(-2)>=64
    return [lq, gt]



def collate_fn_detection(batch):
    """Custom collate function for detection tasks."""

    lq_images = []
    gt_images = []
    batch_idx_list = []
    cls_list = []
    bboxes_list = []

    for i, sample in enumerate(batch):
        lq_images.append(sample['lq'])
        gt_images.append(sample['gt'])

        # Force cls and bboxes into a consistent 2D tensor format.
        cls_i = _ensure_2d_tensor(sample.get('cls', None), 1, dtype=torch.float32)
        bboxes_i = _ensure_2d_tensor(sample.get('bboxes', None), 4, dtype=torch.float32)

        # Check that each class entry has a matching box entry.
        if cls_i.shape[0] != bboxes_i.shape[0]:
            raise ValueError(
                f"Mismatched numbers of cls and bboxes: cls={cls_i.shape}, bboxes={bboxes_i.shape}"
            )

        num_obj = cls_i.shape[0]

        if num_obj > 0:
            batch_idx_i = torch.full((num_obj, 1), i, dtype=torch.float32)
            batch_idx_list.append(batch_idx_i)
            cls_list.append(cls_i)
            bboxes_list.append(bboxes_i)

    lq_images = torch.stack(lq_images, dim=0)   # [B, C, H, W]
    gt_images = torch.stack(gt_images, dim=0)   # [B, C, H, W]

    batch_idx = torch.cat(batch_idx_list, dim=0) if batch_idx_list else torch.zeros((0, 1), dtype=torch.float32)
    cls = torch.cat(cls_list, dim=0) if cls_list else torch.zeros((0, 1), dtype=torch.float32)
    bboxes = torch.cat(bboxes_list, dim=0) if bboxes_list else torch.zeros((0, 4), dtype=torch.float32)

    return {
        'lq': lq_images,            # [B, C, H, W]
        'gt': gt_images,            # [B, C, H, W]
        'batch_idx': batch_idx,     # [total_objects, 1]
        'cls': cls,                 # [total_objects, 1]
        'bboxes': bboxes,           # [total_objects, 4]，normalized YOLO xywh
    }