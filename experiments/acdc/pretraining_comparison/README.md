# ACDC no-augmentation pretraining comparison

Run `../notebooks/08_noaug_swin_voco.ipynb` on the Linux CUDA server in a Python 3.11 venv kernel. Copy the whole checkout and prepared sibling `acdc-data`, including `runs/acdc-aug250-v1-none` (or configure its actual ID). No local training or downloads are needed.

Jobs are independent: run notebook sections 1–3, then **only the chosen launch cell**:
- **4A:** existing U-Net final/best TTA-OFF re-evaluation (requires historical run/checkpoints).
- **4B:** Swin scratch smoke, training and final/best evaluation.
- **4C:** VoCo download, smoke, training and final/best evaluation.

Another instance can skip 4A and 4B entirely. Swin jobs need prepared ACDC data and the same patient split/plans, but no historical U-Net checkpoint or other Swin result. Weight download is deferred until 4C. Set the selected cell's START flag to True; set only its RESUME flag to True when resuming. Each job has its own tmux session/config/snapshot/status/log. There is no automatic next-job queue. A GPU lock prevents concurrent jobs on the same GPU.

| Condition | Existing U-Net none | Swin scratch | Swin VoCo |
|---|---|---|---|
| New training | No | Yes | Yes |
| Deep supervision during training | ON | OFF | OFF |
| Random augmentation / mirror TTA | OFF / OFF in new evaluation | OFF / OFF | OFF / OFF |
| Initialization | Historical random weights | Seed 12345 | Same initialization, then complete Swin encoder overwritten |
| Framework | nnU-Net v2 | nnU-Net v2; MONAI network only | Same |

Common: fold 0, 160 training/40 validation ED/ES volumes, 250 epochs with 250 training and 50 validation steps, batch 4, AMP, nnU-Net SGD (LR .01, momentum .99, Nesterov, weight decay 3e-5), PolyLR, Dice+CE, unchanged ACDC plans/preprocessing and patch sampling/foreground oversampling. No gamma, mirror, geometric, intensity or quality augmentation. No augmentation does not remove crop/padding or sampling. Swin dropout is zero. Seed does not guarantee bitwise reproducibility of multiprocessing or CUDA, and the historical U-Net was not trained with this fixed seed.

Swin uses MONAI 1.5.1 `use_v2=True`, feature48, depths 2/2/2/2, heads 3/6/12/24, gradient checkpointing. The existing `[20,256,224]` patch is padded internally to `[32,256,224]`, then logits are cropped back. Do not reduce batch/patch silently on OOM: this would break the fixed historical comparison. The server smoke determines feasibility; it is not a guarantee against all later CUDA allocations.

## Pretrained release and transfer policy

Official `Luffy503/VoCo/VoCo_B_SSL_head.pt` (Base SSL, Large-Scale-Medical / PreCT-160K extension, **not** the original CVPR checkpoint). Only that file is downloaded (about 220 MB). The Hugging Face commit, LFS SHA256 and source URL are stored in the suite; checksum and size are verified. A complete `swinViT` encoder is required, with known module/name prefix normalization. Missing keys or incompatible shapes stop execution. SSL heads and all non-encoder weights are ignored and listed in `voco-load-report.json`; decoder/output initialization remains random. All parameters are trainable. `torch.load(weights_only=True)` is used; there is no unsafe fallback.

Sources:
- https://github.com/Luffy03/Large-Scale-Medical
- https://github.com/Luffy03/VoCo
- https://huggingface.co/Luffy503/VoCo

This is an encoder-transfer comparison under the nnU-Net recipe, not reproduction of VoCo's official downstream optimization. Existing U-Net DS ON versus Swin DS OFF remains a confound, explicitly recorded in every comparison row.

## Outputs and resume

`acdc-data/pretraining_suites/<SUITE_ID>/jobs/<unet_existing|scratch|voco>/` contains each job's `config.json`, `snapshot.json`, `runner.log`, `status.json`, and (for Swin) `smoke.json`. VoCo also records the encoder load report. Evaluations live in the suite's `evaluation_no_tta/<model>/<final|best>/`.

New checkpoints/logs/progress plots remain in `acdc-data/runs/<SUITE_ID>-<scratch|voco>/nnUNet_results/.../fold_0/`. Historical weights/snapshots/TTA predictions remain intact.

To merge different instances, copy their evaluation folders under separate directories on the analysis server and list the parent directories (containing `evaluation_no_tta`) in section 6's `RESULT_ROOTS`. **Only summary.json files are required for the comparison**; retain manifest.json/case-metrics.csv for provenance. No large checkpoint/prediction transfer is required. The merger validates path-independent split/plans/fingerprint/dataset hashes, metric-code hash and common protocol, rejects incompatible or conflicting duplicate results, and permits partial results. Section 6 writes comparison.json/csv/png under the local suite. Model-level DS differences remain explicit.

Use the same SUITE_ID/seed on both servers, adjusting local storage path/GPU if necessary. Code/config changes for an already frozen job require a new suite ID. Other job configs are independent. This supports separate fresh jobs on different instances; relocating and resuming an already trained run remains subject to the existing strict training provenance checks.
CPU-only checks: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/acdc/tests -p test_pretraining_comparison.py -v`. GPU, real checkpoint compatibility and full-volume inference must be validated by the notebook's server checks.
