"""ACDC controlled comparison: nnU-Net recipe, no deep supervision, 250 epochs.

MONAI supplies only the Swin network. nnU-Net owns data, loss, optimization and inference.
"""
import torch
from torch import nn
from torch.nn import functional as F

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)


class PaddedSwinUNETRV2(nn.Module):
    """Keep nnU-Net's field of view; pad symmetrically for Swin and crop logits back."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Lazy import: existing nnU-Net experiments need not install MONAI.
        from monai.networks.nets import SwinUNETR
        self.network = SwinUNETR(
            in_channels=in_channels, out_channels=out_channels, spatial_dims=3,
            feature_size=24, depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
            patch_size=2, window_size=7, mlp_ratio=4.0, qkv_bias=True,
            norm_name='instance', normalize=True, patch_norm=False,
            drop_rate=0.0, attn_drop_rate=0.0, dropout_path_rate=0.0,
            downsample='merging', use_v2=True, use_checkpoint=True,
        )

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError('Expected [batch, channels, z, y, x]')
        shape = x.shape[2:]
        extra = [(-s) % 32 for s in shape]
        before = [n // 2 for n in extra]
        padding = [v for n, low in reversed(list(zip(extra, before))) for v in (low, n - low)]
        logits = self.network(F.pad(x, padding, mode='constant', value=0))
        return logits[:, :, before[0]:before[0] + shape[0],
                      before[1]:before[1] + shape[1], before[2]:before[2] + shape[2]]


class nnUNetTrainerACDCUNetNoDS250(nnUNetTrainerNoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250

    def set_deep_supervision_enabled(self, enabled: bool):
        # Base validation unconditionally calls this with True on exit. Ignore that
        # request: these networks and their targets must stay single-output forever.
        return None


class nnUNetTrainerACDCSwinV2NoDS250(nnUNetTrainerACDCUNetNoDS250):
    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision=True):
        if enable_deep_supervision:
            raise ValueError('Swin comparison requires deep supervision=False')
        return PaddedSwinUNETRV2(num_input_channels, num_output_channels)
