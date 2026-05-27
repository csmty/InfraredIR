import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import cv2
import types
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics import YOLO

from nets.segformer.test import load_model
import torch.nn.functional as F

class Hyperparameters:
    def __init__(self, box=7.5, cls=0.5, dfl=1.5):
        self.box = box
        self.cls = cls
        self.dfl = dfl

class YOLOv8LossCalculator:
    def __init__(self, model_path, hyp=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(device)

        # Load YOLO arguments and convert them to an attribute-accessible object if needed.
        raw_args = getattr(self.model, 'args', None)

        if raw_args is None:
            self.hyp = types.SimpleNamespace()
        elif isinstance(raw_args, dict):
            self.hyp = types.SimpleNamespace(**raw_args)
        else:
            self.hyp = raw_args

        # Override default hyperparameters with user-defined values if provided.
        if hyp is not None:
            for key, value in hyp.items():
                setattr(self.hyp, key, value)

        # Set default detection loss weights when they are not available.
        if not hasattr(self.hyp, 'box'):
            self.hyp.box = 7.5
        if not hasattr(self.hyp, 'cls'):
            self.hyp.cls = 0.5
        if not hasattr(self.hyp, 'dfl'):
            self.hyp.dfl = 1.5

        self.model.args = self.hyp
        self.model.model.args = self.hyp

        self.criterion = v8DetectionLoss(self.model.model)

        self.criterion.hyp = self.hyp
    
        self.model.eval()
        self.model.model.eval()
        for p in self.model.model.parameters():
            p.requires_grad_(False)

    def _move_targets_to_device(self, batch_targets):
        out = {}
        for k, v in batch_targets.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            else:
                out[k] = torch.as_tensor(v, device=self.device)
        return out

    def calculate_batch_loss(self, batch_images, batch_targets):
        """
        Compute the detection loss for a batch and return a scalar loss
        that can be directly used for backpropagation.
        """
        batch_images = batch_images.to(self.device).float().clamp(0, 1)
        batch_targets = self._move_targets_to_device(batch_targets)

        preds = self.model.model(batch_images)
        total_loss, loss_items = self.criterion(preds, batch_targets)

        # Some Ultralytics versions return a scalar loss,
        # while others may return a loss vector.
        if isinstance(total_loss, torch.Tensor) and total_loss.ndim > 0:
            total_loss = total_loss.sum()

        return total_loss

class SegFormerLossCalculator:
    def __init__(self, cfg, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = load_model(cfg, model_path)
        self.model.to(device)
        self.model.eval()

        # Freeze segmentation model parameters and use it only as a task loss network.
        for param in self.model.parameters():
            param.requires_grad = False

        # print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.criterion = self.criterion.to('cuda')
        
    def calculate_batch_loss(self, batch_images, batch_targets=None):
        """
        Args:
            batch_images: Input images with shape [B, C, H, W].
            batch_targets: Segmentation labels with shape [B, 1, H, W].

        Returns:
            loss: Segmentation loss.
            output: Predicted segmentation map with shape [B, 1, H, W].
        """

        output = self.model(batch_images)

        # Resize logits back to the input resolution.
        _, _, H, W = batch_images.shape
        output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=True)
        
        labels = batch_targets.squeeze(1).long()
        loss = self.criterion(output, labels)

        # Convert logits to discrete prediction masks.
        output = F.softmax(output, dim=1)
        output = torch.argmax(output, dim=1, keepdim=True)  # (B, 1, H, W)
        
        return loss, output
    

