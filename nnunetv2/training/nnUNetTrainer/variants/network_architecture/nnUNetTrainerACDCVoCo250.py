"""ACDC no-augmentation Swin control/VoCo; nnU-Net owns training and inference."""
import json
import faulthandler
import sys
import time
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerACDCNoDS250 import nnUNetTrainerACDCUNetNoDS250
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerACDCTheme250 import nnUNetTrainerACDCNoAug250


def select_encoder_weights(source, target):
    """Require every Swin encoder parameter/buffer; never silently accept partial loading."""
    if 'state_dict' in source:
        source = source['state_dict']
    normalized = {}
    for key, value in source.items():
        key = key.removeprefix('module.').replace('swin_vit.', 'swinViT.')
        if key in normalized:
            raise ValueError(f'Duplicate checkpoint key: {key}')
        normalized[key] = value
    required = {k: v for k, v in target.items() if k.startswith('swinViT.')}
    if not required:
        raise ValueError('Network has no Swin encoder')
    missing = [k for k in required if k not in normalized]
    mismatched = [k for k in required if k in normalized and normalized[k].shape != required[k].shape]
    if missing or mismatched:
        raise ValueError(f'VoCo encoder incompatible: missing={missing}, shape_mismatch={mismatched}')
    # Deliberately transfer encoder only. Decoder/output initialization stays identical to scratch.
    return {k: normalized[k] for k in required}, sorted(set(normalized) - set(required))


def load_voco(network, checkpoint):
    source = torch.load(checkpoint, map_location='cpu', weights_only=True)
    selected, ignored = select_encoder_weights(source, network.network.state_dict())
    network.network.load_state_dict(selected, strict=False)
    return {'policy': 'complete swinViT encoder only; decoder and output randomly initialized',
            'loaded_keys': sorted(selected), 'ignored_source_keys': ignored,
            'loaded_elements': sum(v.numel() for v in selected.values())}


class PaddedSwinBase(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        from monai.networks.nets import SwinUNETR
        self.network = SwinUNETR(
            in_channels=in_channels, out_channels=out_channels, spatial_dims=3,
            feature_size=48, depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
            patch_size=2, window_size=7, mlp_ratio=4., qkv_bias=True,
            norm_name='instance', normalize=True, patch_norm=False,
            drop_rate=0., attn_drop_rate=0., dropout_path_rate=0.,
            downsample='merging', use_v2=True, use_checkpoint=True)

    def forward(self, x):
        shape = x.shape[2:]
        extra = [(-s) % 32 for s in shape]
        before = [n // 2 for n in extra]
        padding = [v for n, low in reversed(list(zip(extra, before))) for v in (low, n-low)]
        y = self.network(F.pad(x, padding))
        return y[:, :, before[0]:before[0]+shape[0], before[1]:before[1]+shape[1], before[2]:before[2]+shape[2]]


class nnUNetTrainerACDCSwinScratchNoAug250(nnUNetTrainerACDCUNetNoDS250):
    use_voco = False
    # Bound to the existing no-augmentation class; sampling and crop remain unchanged.
    get_training_transforms = staticmethod(nnUNetTrainerACDCNoAug250.get_training_transforms)

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        seed = int(os.environ.get('ACDC_SEED', '12345'))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        super().__init__(plans, configuration, fold, dataset_json, device)

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision=True):
        if enable_deep_supervision:
            raise ValueError('This comparison requires DS OFF')
        return PaddedSwinBase(num_input_channels, num_output_channels)

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        result = super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        self.inference_allowed_mirroring_axes = None
        return (*result[:3], None)

    def _phase(self, message):
        self.print_to_log_file(message)
        sys.stdout.flush()

    def get_dataloaders(self):
        self._phase('[startup] Waiting for train/validation loaders and their first batches')
        loaders = super().get_dataloaders()
        self._phase('[startup] Both loaders ready')
        return loaders

    def plot_network_architecture(self):
        # hiddenlayer tracing is optional and can run expensive GPU forwards before epoch 0.
        self._phase('[startup] Skipped optional hiddenlayer network graph')

    def _save_debug_information(self):
        self._phase('[startup] Saving debug information')
        super()._save_debug_information()
        self._phase('[startup] Debug information saved')

    def on_train_start(self):
        self._phase('[startup] Preparing training; stack trace every 120s until ready')
        faulthandler.dump_traceback_later(120, repeat=True)
        try:
            super().on_train_start()
            if self.use_voco and self.current_epoch == 0:
                checkpoint = os.environ.get('ACDC_VOCO_CHECKPOINT')
                if not checkpoint:
                    raise RuntimeError('ACDC_VOCO_CHECKPOINT is required for fresh VoCo training')
                self._phase('[startup] Loading VoCo encoder')
                report = load_voco(self.network, checkpoint)
                (Path(self.output_folder) / 'voco-load-report.json').write_text(json.dumps(report, indent=2))
                self.print_to_log_file('Loaded complete VoCo Swin encoder:', report['loaded_elements'])
            self._phase('[startup] Ready for epoch loop')
        finally:
            faulthandler.cancel_dump_traceback_later()

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._progress_steps = 0
        self._progress_started = time.monotonic()
        self._phase(f'[train] Epoch {self.current_epoch}: waiting for first batch')

    def train_step(self, batch):
        if self._progress_steps == 0:
            self._phase('[train] First batch received; starting forward/backward')
        result = super().train_step(batch)
        self._progress_steps += 1
        if self._progress_steps == 1 or self._progress_steps % 25 == 0:
            elapsed = time.monotonic() - self._progress_started
            self._phase(f'[train] Epoch {self.current_epoch}: '
                        f'{self._progress_steps}/{self.num_iterations_per_epoch} steps; {elapsed:.1f}s')
        return result


class nnUNetTrainerACDCSwinVoCoNoAug250(nnUNetTrainerACDCSwinScratchNoAug250):
    use_voco = True
