"""ACDC fold experiments: one augmentation theme, otherwise base trainer defaults.

All variants use the same 250-epoch schedule and retain base inference mirroring.
SpatialTransform remains for final cropping even when rotation/scaling are disabled.
"""
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform


class nnUNetTrainerACDCTheme250(nnUNetTrainer):
    augmentation_theme = None

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        if self.augmentation_theme not in ('spatial', 'quality', 'intensity', 'none'):
            raise ValueError('Choose a concrete ACDC augmentation theme trainer')
        self.num_epochs = 250

    @classmethod
    def get_training_transforms(cls, patch_size, rotation_for_DA, deep_supervision_scales,
                                mirror_axes, do_dummy_2d_data_aug, use_mask_for_norm=None,
                                is_cascaded=False, foreground_labels=None, regions=None,
                                ignore_label=None):
        if is_cascaded or regions is not None or ignore_label is not None:
            raise ValueError('These trainers support non-cascade, ordinary-label ACDC only')
        spatial = cls.augmentation_theme == 'spatial'
        transforms = []
        if do_dummy_2d_data_aug:
            transforms.append(Convert3DTo2DTransform())
        transforms.append(SpatialTransform(
            patch_size[1:] if do_dummy_2d_data_aug else patch_size,
            patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
            p_rotation=0.2 if spatial else 0, rotation=rotation_for_DA,
            p_scaling=0.2 if spatial else 0, scaling=(0.7, 1.4),
            p_synchronize_scaling_across_axes=1, bg_style_seg_sampling=False,
            border_mode_seg='constant', padding_value_seg=-1))
        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        if cls.augmentation_theme == 'quality':
            transforms.extend([
                RandomTransform(GaussianNoiseTransform(
                    noise_variance=(0, 0.1), p_per_channel=1,
                    synchronize_channels=True), apply_probability=0.1),
                RandomTransform(GaussianBlurTransform(
                    blur_sigma=(0.5, 1.), synchronize_channels=False,
                    synchronize_axes=False, p_per_channel=0.5,
                    benchmark=True), apply_probability=0.2),
                RandomTransform(SimulateLowResolutionTransform(
                    scale=(0.5, 1), synchronize_channels=False, synchronize_axes=True,
                    ignore_axes=(0,) if do_dummy_2d_data_aug else None,
                    allowed_channels=None, p_per_channel=0.5), apply_probability=0.25),
            ])
        if cls.augmentation_theme == 'intensity':
            transforms.extend([
                RandomTransform(MultiplicativeBrightnessTransform(
                    multiplier_range=BGContrast((0.75, 1.25)),
                    synchronize_channels=False, p_per_channel=1), apply_probability=0.15),
                RandomTransform(ContrastTransform(
                    contrast_range=BGContrast((0.75, 1.25)), preserve_range=True,
                    synchronize_channels=False, p_per_channel=1), apply_probability=0.15),
            ])
        if spatial and mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))
        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i, value in enumerate(use_mask_for_norm) if value],
                channel_idx_in_seg=0, set_outside_to=0))
        transforms.append(RemoveLabelTansform(-1, 0))
        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        return ComposeTransforms(transforms)


class nnUNetTrainerACDCSpatial250(nnUNetTrainerACDCTheme250):
    augmentation_theme = 'spatial'


class nnUNetTrainerACDCQuality250(nnUNetTrainerACDCTheme250):
    augmentation_theme = 'quality'


class nnUNetTrainerACDCIntensity250(nnUNetTrainerACDCTheme250):
    augmentation_theme = 'intensity'


class nnUNetTrainerACDCNoAug250(nnUNetTrainerACDCTheme250):
    """Optional control: keep patch sampling/cropping, disable random augmentations."""
    augmentation_theme = 'none'
