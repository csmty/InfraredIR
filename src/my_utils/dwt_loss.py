import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

class DWTLoss(nn.Module):
    def __init__(self, wave='haar', mode='symmetric', level=1, device=None):
        """
        Args:
            wave: Wavelet type, e.g. 'haar'.
            mode: Padding mode used in DWT.
            level: Number of decomposition levels.
        """
        super(DWTLoss, self).__init__()
        self.level = level
        # DWTForward returns:
        #   yl: low-frequency component
        #   yh: list of high-frequency components, each with shape [B, C, 3, H, W]
        self.dwt = DWTForward(J=level, wave=wave, mode=mode)

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image, shape [B, C, H, W].
            target: Ground-truth image, shape [B, C, H, W].

        Returns:
            Total DWT-domain loss over low- and high-frequency components.
        """
        # Apply DWT to prediction and target
        yl_pred, yh_pred = self.dwt(pred)
        yl_target, yh_target = self.dwt(target)

        # Low-frequency reconstruction loss
        # loss_ll = F.mse_loss(yl_pred, yl_target, reduction='mean')

        # High-frequency detail loss across all decomposition levels
        loss_detail = 0.0
        for j in range(len(yh_pred)):
            for i in range(yh_pred[j].shape[2]):  # three detail sub-bands: LH, HL, HH
                # loss_detail += F.l1_loss(yh_pred[j][:, :, i, :, :], yh_target[j][:, :, i, :, :])
                loss_detail += F.mse_loss(yh_pred[j][:, :, i, :, :], yh_target[j][:, :, i, :, :], reduction='mean')

        # total_loss = loss_ll + loss_detail
        total_loss = loss_detail
        return total_loss