"""AMOS MRI: fixed 250 epochs, deep supervision, augmentation and precision ablations.

Training/validation steps follow the repository base trainer, with explicit precision.
The custom predictor uses FP32 CPU accumulation to bound GPU memory usage.
"""
from contextlib import nullcontext
from unittest.mock import patch
import importlib
import os
import random
import numpy as np
import torch
from torch import autocast
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform


class AMOSPredictor(nnUNetPredictor):
    use_amp = False

    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, input_image):
        self.network.to(self.device).eval()
        data, undo = pad_nd_image(input_image.float().cpu(), self.configuration_manager.patch_size,
                                 'constant', {'value': 0}, True, None)
        slices = self._internal_get_sliding_window_slicers(data.shape[1:])
        logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]), dtype=torch.float32)
        weights = torch.zeros(data.shape[1:], dtype=torch.float32)
        gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1/8,
                                    value_scaling_factor=10, dtype=torch.float32, device=torch.device('cpu')) if self.use_gaussian else 1
        for sl in slices:
            x = data[sl][None].to(self.device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=self.use_amp) if self.device.type == 'cuda' else nullcontext():
                prediction = self._internal_maybe_mirror_and_predict(x)[0].float().cpu()
            logits[sl] += prediction * gaussian
            weights[sl[1:]] += gaussian
        logits /= weights
        if not torch.isfinite(logits).all():
            raise RuntimeError('Non-finite sliding-window predictions')
        return logits[(slice(None), *undo[1:])]


class AMOSAMPPredictor(AMOSPredictor):
    use_amp = True


class nnUNetTrainerAMOSAugFP32(nnUNetTrainer):
    use_amp = False

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        seed = int(os.environ.get('AMOS_SEED', '12345'))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        self.enable_deep_supervision = True
        if not self.use_amp:
            self.grad_scaler = None

    def perform_actual_validation(self, save_probabilities=False):
        # Only this dedicated training process uses the replacement, restored on exit.
        module = importlib.import_module('nnunetv2.training.nnUNetTrainer.nnUNetTrainer')
        predictor = AMOSAMPPredictor if self.use_amp else AMOSPredictor
        with patch.object(module, 'nnUNetPredictor', predictor):
            return super().perform_actual_validation(save_probabilities)

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, dtype=torch.float32, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast('cuda', dtype=torch.float16, enabled=self.use_amp) if self.device.type == 'cuda' else nullcontext():
            output = self.network(data)
            # del data
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, dtype=torch.float32, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast('cuda', dtype=torch.float16, enabled=self.use_amp) if self.device.type == 'cuda' else nullcontext():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}


class nnUNetTrainerAMOSAugAMP(nnUNetTrainerAMOSAugFP32):
    use_amp = True


class nnUNetTrainerAMOSNoAugFP32(nnUNetTrainerAMOSAugFP32):
    @staticmethod
    def get_training_transforms(patch_size, rotation_for_DA, deep_supervision_scales,
                                mirror_axes, do_dummy_2d_data_aug, use_mask_for_norm=None,
                                is_cascaded=False, foreground_labels=None, regions=None, ignore_label=None):
        if is_cascaded or regions is not None or ignore_label is not None:
            raise ValueError('This experiment supports ordinary AMOS labels, 3d_fullres only')
        transforms = []
        if do_dummy_2d_data_aug:
            transforms.append(Convert3DTo2DTransform())
        transforms.append(SpatialTransform(
            patch_size[1:] if do_dummy_2d_data_aug else patch_size,
            patch_center_dist_from_border=0, random_crop=False,
            p_elastic_deform=0, p_rotation=0, rotation=rotation_for_DA,
            p_scaling=0, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
            bg_style_seg_sampling=False, border_mode_seg='constant', padding_value_seg=-1))
        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())
        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i,v in enumerate(use_mask_for_norm) if v],
                channel_idx_in_seg=0, set_outside_to=0))
        transforms.append(RemoveLabelTansform(-1, 0))
        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        return ComposeTransforms(transforms)


class nnUNetTrainerAMOSNoAugAMP(nnUNetTrainerAMOSNoAugFP32):
    use_amp = True
