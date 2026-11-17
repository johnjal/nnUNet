# nnU-Net의 자동 결정 항목: 현재 코드와 ACDC 적용 결과

분석일: 2026-11-17. 현재 checkout HEAD: `37fba7e88b0598abff7dcc9c2c3e2957c7f363b9`.

범위는 이번 실험의 **ExperimentPlanner → DefaultPreprocessor → nnUNetTrainer → 기본 predictor** 경로와, 여기서 분기하는 region/cascade/DDP/모델 선택 기능입니다. 별도로 선택해야 하는 모든 trainer variant의 개별 설정이나 외부 라이브러리 내부 최적화까지 망라한 목록은 아닙니다. 현재 구현의 규칙과 과거 실험의 저장된 plans/debug/log를 구분했습니다. 과거 학습 코드 전체가 현재 checkout과 동일하다고 가정하지 않습니다.

## 먼저 구분해야 하는 네 가지

| 분류 | 의미 | 예 |
|---|---|---|
| 데이터·메타데이터 기반 규칙 | 데이터를 측정하거나 사용자 메타데이터를 읽어 값 결정 | spacing, patch, normalization, region용 loss 분기 |
| 고정 기본값의 자동 적용 | 사용자가 쓰지 않아도 이미 지정된 설정 실행 | SGD, 1,000 epoch, 증강 확률 |
| 실행 중 무작위 선택 | 고정 범위·확률 내에서 매 샘플마다 선택 | 환자/패치 위치, 회전각, 밝기 변화 |
| 검증 결과 기반 선택 | 실제 평가 점수를 비교해 선택 | best checkpoint, 별도 실행하는 configuration/후처리 비교 |

**증강을 자동 실행한다는 것과 데이터마다 최적 증강 정책을 학습한다는 것은 다릅니다. 기본 nnU-Net은 AutoAugment 탐색을 하지 않습니다.**

## 1. 데이터 분석과 계획 생성

주요 진입점: [plan_and_preprocess_api.py](../../../nnunetv2/experiment_planning/plan_and_preprocess_api.py), [fingerprint_extractor.py](../../../nnunetv2/experiment_planning/dataset_fingerprint/fingerprint_extractor.py), [default_experiment_planner.py](../../../nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py).

| 자동 처리/결정 항목 | 코드의 기준 | 이번 ACDC |
|---|---|---|
| 영상 reader/writer | dataset.json의 명시적 override 우선, 그렇지 않으면 파일 확장자·읽기 가능 여부 | SimpleITKIO |
| 데이터 fingerprint | 사례별 spacing, nonzero crop 후 shape, crop 상대 크기; GT foreground의 채널별 강도 통계 | 200개 3D 영상의 통계 |
| 강도 통계 샘플링 | 전 foreground를 저장하는 대신 총 목표 10^8개 샘플을 사례별로 배분; foreground에서 추출 | 통계 수집은 되었지만 MRI Z-score의 평균/표준편차는 각 영상에서 계산 |
| 목표 spacing | 기본 각 축 중앙값; 간격과 voxel 수 양쪽에서 비등방성이 크면 저해상도 축의 10백분위 간격 사용 | 중앙 `[10,1.5625,1.5625]` → 목표 `[5,1.5625,1.5625]` mm |
| 축 순서 | 가장 큰 목표 spacing의 축을 먼저 배치, 역변환도 기록 | transpose `[0,1,2]`, 변화 없음 |
| 채널별 정규화 종류 | 사용자가 제공한 channel name 매핑. CT, nonorm, zscore, rescale_to_0_1, rgb_to_0_1 지원; 매핑 없으면 Z-score | ZScoreNormalization. 영상을 보고 MRI/CT를 분류한 것이 아님 |
| nonzero mask로 정규화할지 | crop 후 상대 크기의 중앙값 < 0.75이고 해당 normalizer가 지원하면 사용 | false |
| 2D/3D 구성 후보 | 2D 계획 생성; transpose 후 첫 축의 대표 크기가 1이 아니면 3D fullres도 생성 | 2d와 3d_fullres 생성, 실제 학습은 사용자가 선택한 3d_fullres |
| lowres/cascade 필요 여부 | fullres patch가 대표 영상 voxel 수의 25% 미만이면 lowres 시도; spacing을 점진 조정; 원본과 크기 차이가 2배 미만이면 lowres 제거 | lowres/cascade 미생성 |
| patch의 초기 비율 | spacing의 역수로 축 비율 구성 후 대표 영상 크기로 제한 | 최종 `[20,256,224]` |
| patch 크기 최종 조정 | downsampling 배수에 맞춤; 추정 메모리가 예산을 초과하면 축을 줄이고 구조 재계산 | `[20,256,224]` |
| batch size | 추정 메모리 여유로 계산, 최소 2; 한 batch가 데이터셋 추정 voxel 수의 5%를 넘지 않게 제한 | 4 |
| batch Dice 여부 | 2D=true, lowres=false, lowres 없는 3D fullres=false; lowres가 있는 fullres=true | **false**: loss의 Dice는 batch 전체 TP를 합쳐 계산하는 모드가 아님 |

메모리 계산은 `compute_conv_feature_map_size`를 이용한 추정입니다. GPU 실측 autotuning이나 하이퍼파라미터 탐색이 아닙니다. 현재 기본 planner의 목표는 **8GB**이며 사용자가 override할 수 있습니다. 이번 실행 wrapper는 memory target 인자를 전달하지 않습니다. 3090의 실제 VRAM 24GB를 자동 탐지해서 전부 활용하도록 만드는 과정이 아닙니다.

`batch_dice=false`와 학습 중 **pseudo-Dice가 validation patch 전체 TP/FP/FN을 합산한다**는 사실은 서로 다른 계산 경로입니다.

## 2. 자동 구성되는 네트워크 요소

근거: planner의 `get_plans_for_configuration`, [network_topology.py](../../../nnunetv2/experiment_planning/experiment_planners/network_topology.py), [get_network_from_plans.py](../../../nnunetv2/utilities/get_network_from_plans.py), [label_handling.py](../../../nnunetv2/utilities/label_handling/label_handling.py).

| 항목 | 결정 방식 | ACDC |
|---|---|---|
| 네트워크 계열 | 기본 planner가 PlainConvUNet으로 고정. Transformer/ResNet 중 성능 비교 후 선택하지 않음 | PlainConvUNet |
| 입력 채널 수 | dataset.json의 채널 수; cascade이면 이전 단계 foreground 채널 추가 | 1 |
| 출력 채널 수 | label 또는 region 정의로 결정 | 배경 포함 4 |
| Conv2d/Conv3d, InstanceNorm2d/3d | 구성의 차원에 따라 대응 클래스 선택 | Conv3d, InstanceNorm3d |
| 단계 수·축별 downsampling 횟수 | spacing 비율과 feature map 크기를 보며 구성; 최소 feature map edge 4 등 규칙 | 6 stages |
| 단계별 stride | 비슷한 유효 spacing과 충분한 크기를 가진 축을 우선 축소 | `[1,1,1]`, `[1,2,2]`, `[2,2,2]`, `[2,2,2]`, `[1,2,2]`, `[1,2,2]` |
| 단계별 kernel | 축 간 spacing 비가 작아지면 1→3; bottleneck은 3 | 첫 `[1,3,3]`, 나머지 `[3,3,3]` |
| 단계별 feature 수 | 32부터 단계별 2배, 3D 최대 320 / 2D 최대 512 | `[32,64,128,256,320,320]` |
| stage당 convolution 수 | 기본 encoder/decoder 모두 2로 고정 | 2 |
| 나머지 구조 기본값 | bias=true, InstanceNorm eps=1e-5·affine=true, LeakyReLU, dropout 없음 | 그대로 |
| 가중치 초기화 | 생성한 network가 제공하는 initialize를 호출; 필요 시 사용자가 pretrained weights 제공 | 초기화 후 학습. 새 초기화 전략을 탐색하지 않음 |

## 3. 사례별 전처리에서 자동 계산되는 것

근거: [default_preprocessor.py](../../../nnunetv2/preprocessing/preprocessors/default_preprocessor.py)의 `run_case_npy`, [cropping.py](../../../nnunetv2/preprocessing/cropping/cropping.py), [default_resampling.py](../../../nnunetv2/preprocessing/resampling/default_resampling.py), [default_normalization_schemes.py](../../../nnunetv2/preprocessing/normalization/default_normalization_schemes.py).

1. 영상별 nonzero bounding box를 계산해 image/seg를 함께 crop합니다. **정답 심장 영역만 잘라내는 crop은 아닙니다.** 원래 shape, crop box, spacing을 복원용으로 보존합니다.
2. 계획된 축 순서를 적용하고, 원래 spacing과 목표 spacing으로 각 영상의 새 shape를 계산합니다. 모든 영상을 동일 shape로 resize하는 것은 아닙니다.
3. 리샘플링 전에 정규화합니다. ACDC는 crop된 영상 전체의 평균/표준편차로 Z-score를 계산합니다. CT 경로는 foreground 통계의 0.5/99.5백분위로 clip하고 데이터셋 평균/표준편차를 사용합니다.
4. 원래/목표 spacing의 비등방성으로 저해상도 축을 분리해 리샘플링할지 결정합니다. 기본 data order=3, seg label-aware order=1, separate-axis order=0입니다. 숫자 라벨 자체를 연속 강도로 보간하지 않습니다.
5. 저장 dtype 등을 정합니다. data는 float32, seg는 클래스 값 범위에 따라 int8/int16입니다. 원래 좌표 정보와 present labels도 기록합니다.
6. 현재 checkout은 foreground 샘플링 위치를 별도 압축 저장소에 추출합니다. 위치 저장 방식은 코드 버전에 따라 다릅니다. 이전 구현의 per-case pickle과 혼동하지 마세요.

ACDC에서 전처리 후 대표 shape는 `[18,237,208]`입니다. 단, 실제 각 영상 shape는 다릅니다.

## 4. 분할과 패치 샘플링

근거: [nnUNetTrainer.py](../../../nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py)의 `do_split`, `get_dataloaders`, [data_loader.py](../../../nnunetv2/training/dataloading/data_loader.py)의 `get_bbox`, `generate_train_batch`, [foreground_locations.py](../../../nnunetv2/training/dataloading/foreground_locations.py).

| 항목 | 자동 동작 | 이번 ACDC |
|---|---|---|
| train/val split | 기존 splits_final.json 우선. 없으면 기본 seed=12345의 case 단위 5-fold 생성 | 실험 wrapper가 seed=1234의 **환자 단위** split을 미리 만듦. 일반 기본 분할이 환자 ID를 자동 이해하는 것은 아님 |
| fold 선택 예외 | fold='all'이면 train/val 모두 전체 사례; 범위 밖 fold는 seeded 80:20 대체 | fold 0만 명시적으로 선택, 160 train / 40 val |
| 매 batch의 사례 선택 | loader가 사례를 선택; 기본 sampling_probabilities=None | 같은 영상이 여러 번 등장 가능. 한 epoch에 전체를 한 번씩 방문하지 않음 |
| 일반 patch 위치 | 허용 bounding box 범위 내 무작위 선택 | 영상별 다른 crop/padding 위치 |
| foreground 강제 slot | 고정 oversample=0.33, probabilistic=false. `j >= round(batch_size*(1-.33))`인 slot에 적용 | batch 4이면 `round(2.68)=3`: **마지막 1개 slot, 즉 25%** |
| foreground 위치 선택 | 존재하는 foreground class/region과 해당 voxel 위치 선택; 없으면 일반 random crop | foreground를 포함하도록 표본 선택. 최종 증강 crop 이후 포함을 절대 보장하는 비율은 아님 |
| crop padding | shape 부족 시 자동 pad. data=0, seg=-1, transform에서 -1→0 | 작은 볼륨도 고정 입력 shape 생성 |
| 증강 전 초기 patch 크기 | 회전·확대축소 중 빈 영역을 줄이도록 더 큰 crop 크기를 계산 | **처음 읽는 patch는 최종 `[20,256,224]`보다 클 수 있음**, 공간 증강 후 최종 크기 |
| validation sampling | 별도 val loader에서도 patch 선택·foreground oversampling 사용 | 증강 없이 50 batch; 전체 40개 볼륨 평가와 다름 |

앞선 설명의 “20×256×224를 샘플링한다”는 모델 입력 크기 기준의 요약입니다. 코드에서는 **증강 여유를 둔 초기 crop → 공간 변환 → 최종 patch**라는 단계가 더 있습니다.

## 5. Augmentation: 자동으로 바뀌는 부분과 고정인 부분

근거: trainer `configure_rotation_dummyDA_mirroring_and_inital_patch_size`와 `get_training_transforms`.

### 5.1 데이터/patch 형태로 결정하는 부분

| 항목 | 규칙 | ACDC 결과 |
|---|---|---|
| dummy 2D augmentation | 3D patch에서 `max(patch)/patch[0] > 3` | `256/20=12.8` → **True** |
| 회전 범위 | 3D 일반 ±30°, dummy 2D ±180°; 2D는 patch 종횡비>1.5이면 ±15°, 아니면 ±180° | 평면 내 ±180° 후보 범위 |
| 공간 변환 차원 | dummy이면 z를 채널처럼 취급해 2D 공간 변환 후 3D로 복원 | 슬라이스들을 같은 평면 변환으로 처리. 모델 자체는 계속 3D |
| low-resolution simulation에서 제외할 축 | dummy이면 ignore_axes=(0,) | z축 해상도를 이 증강에서 변경하지 않음 |
| mirroring 가능 축 | 2D=(0,1), 3D=(0,1,2) | z/y/x, inference_allowed_mirroring_axes에도 저장 |
| 초기 crop 크기 | 회전 범위·patch 크기·여유 scale `(0.85,1.25)`로 계산; dummy이면 z 크기 유지 | 자동 계산. 실제 scale 증강 범위 `(0.7,1.4)`와 다르며 코드에도 이 불일치가 주석으로 남아 있음 |

### 5.2 고정 확률·범위에서 매번 무작위 실행하는 부분

아래는 **현재 기본 trainer가 외부 transform 라이브러리에 전달하는 인자**입니다. 적용 확률은 성능을 보고 조정하지 않습니다. 상위 확률과 채널별 확률이 둘 다 있는 증강은 두 조건을 함께 봐야 합니다.

| 증강 | 상위 적용 확률 | 지정된 범위/옵션 |
|---|---:|---|
| 회전 | 0.20 | 위에서 선택한 회전 범위 |
| Scaling | 0.20 | 0.7~1.4, 공간축 scale 동기화 |
| Elastic deformation | **0** | 기본 경로에서 비활성 |
| Gaussian noise | 0.10 | `noise_variance=(0,0.1)`, 채널 확률 1, 채널 동기화 |
| Gaussian blur | 0.20 | sigma 0.5~1.0, 채널 확률 0.5, 채널/축 비동기화 |
| 밝기 배율 | 0.15 | 0.75~1.25, 채널 확률 1 |
| Contrast | 0.15 | 0.75~1.25, preserve_range=true |
| 저해상도 simulation | 0.25 | scale 0.5~1, 채널 확률 0.5, 공간축 동기화, dummy의 z 제외 |
| 반전 후 Gamma | 0.10 | gamma 0.7~1.5, invert=true, 원래 통계 유지 |
| 일반 Gamma | 0.30 | gamma 0.7~1.5, invert=false, 원래 통계 유지 |
| Mirror | transform 항상 등록, 허용 축 내 랜덤 반전 | trainer가 세부 반전 확률을 별도 전달하지 않고 라이브러리 기본값 사용 |

`BGContrast` 등 범위 sampling의 세부 분포와 MirrorTransform 내부 확률은 설치된 batchgeneratorsv2 구현에 위임됩니다. 범위만 보고 모두 단순 uniform이라고 단정하지 않습니다. 공간 증강은 영상/정답에 일관되게, 강도 증강은 영상에 적용됩니다. 위 표는 각 patch에서 모든 변환이 반드시 발생한다는 뜻이 아닙니다.

조건부 처리도 있습니다:

- use_mask_for_norm=true이면 mask 밖 image를 0으로 처리. ACDC는 false.
- cascade이면 이전 단계 segmentation을 one-hot input으로 추가하고, binary morphology(상위 p=0.4, label별 p=0.5, 크기 1~8), 작은 연결성분 제거(상위 p=0.2, label별 p=0.5, 최대 비중 0.15)를 추가. ACDC는 미사용.
- region 학습이면 정답을 region channel로 변환하고 ignore label을 처리. ACDC는 일반 label 분류.
- deep supervision 출력 해상도에 맞춰 정답 마스크들을 생성.
- **validation에는 회전·강도 변환 등의 random augmentation을 적용하지 않습니다.** -1 label 제거, 필요 시 cascade/region 변환, deep-supervision용 정답 생성만 합니다.

## 6. Loss와 학습 동작

근거: trainer `_build_loss`, `_get_deep_supervision_scales`, `configure_optimizers`, `train_step`, [polylr.py](../../../nnunetv2/training/lr_scheduler/polylr.py).

| 항목 | 자동/기본 동작 | ACDC |
|---|---|---|
| loss 종류 | 일반 label이면 Dice+CE; region이면 Dice+BCE. ignore label 유무에 따라 mask 적용 | Dice+CE, 두 항 가중치 1, Dice 배경 제외, smooth=1e-5 |
| batch Dice | plans에서 읽음 | false |
| deep supervision 활성화 | 기본 True | True |
| deep-supervision 해상도 | stride 누적곱의 역수에서 계산 | 6단계 구조에서 5개 decoder 출력 scale |
| deep-supervision 가중치 | 1/2^i로 감소, 최저 출력 가중치 0(DDP는 1e-6), 합이 1이 되도록 정규화 | 단일 GPU 기준 `[8/15,4/15,2/15,1/15,0]` |
| optimizer | 기본 SGD, momentum .99, Nesterov=true, weight decay 3e-5 | 고정값 그대로 |
| 학습률 | initial .01, `lr=.01*(1-epoch/num_epochs)^.9` | 1,000 epoch 일정. val 성능을 보고 LR을 낮추는 방식 아님 |
| 학습 길이 | 기본 1,000 epoch, train 250 step, val 50 step | 그대로. **자동 early stopping 없음** |
| gradient clipping | norm 12 고정 | 적용 |
| mixed precision | CUDA에서 autocast·GradScaler 사용, CPU/MPS에서는 해당 경로 비활성 | CUDA 적용 |
| epoch별 평가 | loss 평균, class별 TP/FP/FN 합계에서 pseudo-Dice 계산 | foreground 3개 클래스 평균 |
| EMA | `0.9*이전 + 0.1*현재 pseudo-Dice`, 첫 값으로 초기화 | 점수의 EMA. 가중치 EMA 아님 |
| checkpoint 선택 | EMA 최고 갱신 시 best, 기본 50 epoch마다 latest, 마지막 final | 학습 종료 후 final을 기본 평가. 최저 loss/최대 mIoU로 best를 고르지 않음 |

## 7. 실행 환경에 따른 자동 분기

근거: trainer `initialize`, `_do_i_compile`, `_set_batch_size_and_oversample`, `get_dataloaders`, [default_n_proc_DA.py](../../../nnunetv2/utilities/default_n_proc_DA.py).

- DDP가 명시적으로 시작된 환경이면 rank별 GPU, batch 분할, foreground 강제 slot 비율, gradient 동기화, validation 사례 분배를 설정합니다. GPU가 두 개 보인다는 이유만으로 DDP를 시작하지는 않습니다.
- compile은 장치/OS/`nnUNet_compile` 환경변수에 따라 분기합니다. 이번 실험은 wrapper에서 **false**로 명시했습니다.
- 증강 worker 수는 환경변수 우선, 아니면 hostname 테이블/기본값과 CPU 수로 제한합니다. 부하를 실측해 최적화하는 알고리즘은 아닙니다. 이번 실험은 train=4, val=2입니다.
- worker 수에 따른 queue cache 크기, CUDA pin_memory, dataset 파일 형식에 맞는 loader를 선택합니다. 데이터 저장 압축/블록 크기 같은 I/O 정책도 구현에 있으나 성능을 학습해 정하는 모델 hyperparameter는 아닙니다.
- 계획된 batch 크기를 학습 중 OOM에 맞춰 계속 줄이는 자동 기능은 이 기본 training loop에 없습니다.

## 8. 전체 볼륨 validation과 inference

근거: trainer `perform_actual_validation`, [predict_from_raw_data.py](../../../nnunetv2/inference/predict_from_raw_data.py), [sliding_window_prediction.py](../../../nnunetv2/inference/sliding_window_prediction.py), [export_prediction.py](../../../nnunetv2/inference/export_prediction.py).

1. checkpoint/plans에서 네트워크 구조·가중치·전처리 설정·허용 mirror 축을 읽습니다. 평가 시 deep supervision을 끕니다.
2. 입력 shape와 patch 크기로 padding 및 sliding-window 위치를 계산합니다. 기본 step size=0.5, overlap은 Gaussian 가중치로 합성합니다.
3. mirroring TTA가 켜져 있으면 허용 축의 반전 조합을 예측하고 되돌려 평균합니다. ACDC 허용 축은 3개입니다. 학습 중 random augmentation과는 다른 과정입니다.
4. fold를 지정하지 않은 predictor 호출은 사용 가능한 fold를 탐지할 수 있습니다. 여러 fold를 지정하면 예측을 합칩니다. 이번 실험은 fold 0 하나이므로 5-fold ensemble이 아닙니다.
5. region 여부에 따라 sigmoid/softmax와 mask 변환을 선택합니다. 일반 label은 argmax, region은 threshold 기반입니다.
6. 모델 출력의 spacing/shape를 되돌리고 crop을 복원하며 transpose도 되돌려 원래 좌표계로 저장합니다. 저장 여부 옵션에 따라 probabilities도 내보냅니다.
7. GPU에 전체 결과를 누적하다 RuntimeError가 발생하면 CPU 결과 누적 경로로 재시도할 수 있습니다. **네트워크 학습 batch 자동 축소와는 다른 기능**입니다.
8. 전체 validation 결과를 GT와 비교해 보고합니다. NSD 계산과 보고서 시각화는 이번 실험 wrapper에서 추가한 기능입니다.

## 9. 결과를 보고 자동 선택하지만 별도 실행이 필요한 기능

근거: [find_best_configuration.py](../../../nnunetv2/evaluation/find_best_configuration.py), [remove_connected_components.py](../../../nnunetv2/postprocessing/remove_connected_components.py).

| 항목 | 선택 기준 | 이번 실험 |
|---|---|---|
| 최종 configuration 선택 | 이미 학습된 후보의 cross-validation foreground mean Dice 비교 | 미실행; 3d_fullres를 미리 선택 |
| configuration ensemble | 허용하면 후보 쌍의 확률 ensemble 평가 후 단독 모델과 비교 | 미실행 |
| 후처리 선택 | 최대 foreground 연결성분만 남겼을 때 평균 Dice 개선·클래스 악화 없음 여부, 이후 class/region별 최대 성분 유지의 개선 여부 평가 | 미실행 |

위 기능은 후보를 자동으로 모두 학습하는 NAS가 아닙니다. 필요한 모델 학습/확률 저장을 먼저 하고 해당 명령을 실행해야 합니다. 일반 predict가 매번 GT를 보고 최선의 후처리를 선택하는 것도 아닙니다.

## 10. 사용자가 결정해야 하는 것 / 우리 wrapper에서 추가한 것

- 데이터셋·사용할 채널·라벨 의미·시간축 포함 여부·환자/기관 단위 분할 정책은 사용자가 정의합니다.
- planner/trainer 종류, 사용할 configuration/fold/GPU, memory target override, 사전학습 가중치 사용 여부, 실험 예산/최종 평가 정책은 사용자 선택입니다.
- 다운로드, 50GB 저장 공간 제약, 파일 checksum 검증, config snapshot, 실행 허용 guard, tmux, GPU 사용량 기록, 결과 CSV/시각화, 04번 확률 loss 분석, 05번 epoch 선택 분석은 **우리 실험 관리 코드가 추가한 기능**입니다.
- 현재 05번에서 250 epoch를 후보로 정한 것은 사람의 분석입니다. nnU-Net이 자동으로 250 epoch를 선택한 것이 아니며 기본 training schedule은 여전히 1,000입니다.

## 실제 실행 증거와 읽는 순서

보관한 [plans](artifacts/runs/acdc-baseline-v1/nnUNet_results/Dataset027_ACDC/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json), [debug](artifacts/runs/acdc-baseline-v1/nnUNet_results/Dataset027_ACDC/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/debug.json), [학습 로그](artifacts/runs/acdc-baseline-v1/nnUNet_results/Dataset027_ACDC/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_log_2026_11_11_03_49_15.txt), [실험 workflow](../scripts/workflow.py)를 대조했습니다.

1. `ExperimentPlanner.determine_fullres_target_spacing` → `get_plans_for_configuration`: 데이터가 어떻게 모델 설정으로 바뀌는가.
2. `nnUNetTrainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size` → `get_training_transforms`: 증강에서 무엇이 규칙이고 무엇이 고정인가.
3. `nnUNetDataLoader.get_bbox` → `generate_train_batch`: 어느 위치를 어떤 크기로 읽는가.
4. `nnUNetTrainer._build_loss` → `train_step` → `on_epoch_end`: 실제 최적화와 checkpoint 선택.
5. `perform_actual_validation` → predictor/export: 패치 평가와 전체 볼륨 평가의 차이.
