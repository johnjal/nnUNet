# ACDC: SwinUNETR V2 / U-Net, no deep supervision

Use `../notebooks/07_swinunetr_v2_no_deep_supervision.ipynb` on the Linux GPU server.
The notebook contains initialization, installation, background launch/resume, monitoring,
train/validation loss curves, final/best comparison and an explicit stop cell.

## Experiment contract

- New random initialization for both models; no pretrained encoder or old baseline checkpoint.
- One existing patient-grouped fold (0): 160 training / 40 validation ED/ES volumes.
- 250 epochs, 250 train + 50 validation iterations per epoch; 62,500 optimizer updates.
- No deep supervision in either network; same nnU-Net Dice+CE, SGD and PolyLR recipe.
- All normal nnU-Net training augmentations, including both gamma transforms; elastic remains off.
- Existing spacing `[5,1.5625,1.5625]`, patch `[20,256,224]`, batch 4 by default.
- Single GPU, AMP, no compilation. No fixed training seed; split is fixed.
- Swin network from MONAI 1.5.1, feature24/depths2222/heads3,6,12,24,
  patch2/window7/merging/use_v2=True; dropout rates zero; gradient checkpointing enabled.
- Symmetric zero-padding inside Swin to multiples of 32 and exact crop of output logits.
  This is our compatibility adapter, not a verified historical paper setting.
- nnU-Net owns preprocessing, training, checkpoint management, sliding-window inference and final evaluation.
  MONAI supplies only the network. The existing U-Net planner is not a Swin memory estimator.
- This compares the architectures under one nnU-Net training recipe. It does not claim
  that SGD .01 is the best Swin-specific optimizer or that all historical paper settings are recovered.

## Server workflow

1. Copy the repository and the **prepared** sibling `acdc-data` folder to the server.
   If no prepared folder exists, finish notebook 01 first. Raw validation images/labels,
   preprocessed arrays, plans, fingerprint, split and metadata/download manifest are required.
2. Select a Python 3.11 venv kernel. Use a separate venv if another experiment still uses the old environment.
3. Set paths and run the installation cell once (`INSTALL_PACKAGES=True`), then restart
   the kernel and set it back to False. PyTorch matches the prior successful cu128 environment;
   a fresh-process CUDA check must pass on the actual server/driver.
4. Set `START_SUITE=True` and launch. The tmux queue executes **Swin → U-Net**.
   It first checks both networks with synthetic data at the full batch/patch size on the GPU.
   Those test weights are discarded before training. No dataset training occurs during smoke.
5. View status/logs. Loss plots and comparison cells can be rerun without starting training again.
6. Interrupted identical configurations use `RESUME=True`. Finished training is skipped;
   valid best prediction masks are reused. Final checkpoint without final validation is resumed
   through nnU-Net to complete validation.

Do not modify code/packages while the queue is active. Configuration, input metadata,
training source, driver files and packages are frozen. A new configuration needs a new suite ID.
OOM does not silently change settings: choose a new ID and common batch 2 (or 1) for **both**
models. Foreground slot rounding changes with batch; this is a documented change to the recipe.
The full-batch GPU smoke is a feasibility check, not a guarantee against every later allocation failure.

## Output layout (defaults)

```text
acdc-data/
  nnUNet_preprocessed/Dataset027_ACDC/nnUNetPlansNoDS250BS4.json
  runs/
    acdc-swinv2-unet-nods250-v1-swin_v2/
    acdc-swinv2-unet-nods250-v1-unet/
  architecture_suites/acdc-swinv2-unet-nods250-v1/
    suite.server.json
    suite.snapshot.json
    status.json
    smoke.json
    runner.log
    comparison.csv
    comparison.json
    comparison-final.png       # written when notebook comparison cell runs
    comparison-best.png
    best/swin_v2/              # masks, manifest, case-metrics.csv, summary.json
    best/unet/
```

Final predictions remain in each run's normal `fold_0/validation`. Best predictions
never overwrite final. No probability arrays are retained. Compare final against final
and best against best. Best selection uses EMA pseudo-Dice, whereas the table reports
whole-volume foreground class-mean Dice and IoU. NSD is not in this table. This validation
set is also used for checkpoint selection and is not an independent test set.

## Files to deploy

Deploy the updated repo, including the new trainer
`nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerACDCNoDS250.py`,
this directory, notebook 07 and `experiments/acdc/analysis/compare_best.py` (metric helpers).
Existing `experiments/acdc/scripts` modules are dependencies. Custom trainer code must
remain importable for future inference and resume.

Adding a trainer changes the source hash used by the older run-freezing system. Do not
attempt to resume old runs against modified source and bypass their checks; keep their
original code/environment when resuming them. Their saved predictions can still be read.

## CPU verification

With repository dependencies and this directory's requirements installed:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python experiments/acdc/tests/test_architecture_comparison.py
```

Tests use temporary synthetic tensors/files only, including Swin V2 backward and
checkpoint reload through nnU-Net inference. They do not run the full-size local model
or real ACDC training. Actual 3090 memory/runtime must be checked on the server.

Sources: [nnU-Net extension documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/extending_nnunet.md),
[MONAI pinned implementation](https://github.com/Project-MONAI/MONAI/blob/1.5.1/monai/networks/nets/swin_unetr.py).
