# SwinUNETR V2의 nnU-Net 통합과 기존 실험 설정 조사

조사일: 2026-11-19. 대상은 nnU-Net Revisited(2404.09556v2)의 nnU-Net 기반 SwinUNETR V2, 공식 확장 문서, 로컬 ACDC baseline 및 네 증강 실험이다. 학습·추론·가중치 다운로드는 수행하지 않았다.

## 결론과 증거 범위

- 논문의 nnU-Net 기반 SwinUNETR V1/V2는 from scratch이다. pretrained encoder를 사용한 실험으로 해석하면 안 된다.
- 논문의 Auto3DSeg SwinUNETR은 사전학습 가중치를 사용하는 별도 실험이다. 공개 BTCV 예제의 SSL 가중치를 논문의 Auto3DSeg가 사용한 정확한 파일이라고 단정할 수 없다.
- 현재 ACDC의 다섯 실험은 PlainConvUNet이다. 실행 경로는 사전학습 가중치를 전달하지 않고 새 모델을 생성하며, resume은 해당 실험 자체의 체크포인트를 이어가는 기능이다. 외부 pretrained encoder는 사용하지 않는 구성이다.
- 공식 nnU-Net 확장 문서는 새 trainer에서 네트워크 생성 함수를 교체하는 방법, deep supervision 미지원 시 NoDeepSupervision trainer 사용, 입력 크기 호환성 및 logits 출력 조건을 명시한다.
- nnU-Net Revisited의 정확한 Swin V2 trainer/commit/전체 하이퍼파라미터는 이번 조사로 확보하지 못했다. 현재 MONAI 기본값이나 다른 연구의 trainer로 그 빈칸을 채우지 않는다.

본문 §3.1의 설정 원칙은 자동 설정 → 유사 데이터 설정 → 제공 기본 설정 → 필요시 학습률 감소이다. 모든 모델에 같은 optimizer와 LR을 강제했다고 쓰여 있지 않다. 따라서 앞서 제안한 SGD 0.01은 **현재 실험과의 통제 비교안**이며 **논문 Swin의 확인된 설정이 아니다**. [논문](https://arxiv.org/html/2404.09556v2)

## 공식 documentation에서 찾은 구현 지침

| 문서 | 확인한 내용 | 이번 실험에 적용 |
|---|---|---|
| [extending_nnunet.md](../../../documentation/extending_nnunet.md) | trainer 상속, build_network_architecture 재정의; DS 미지원 모델은 NoDeepSupervision 기반; 마지막 softmax/sigmoid 금지 | nnU-Net 학습 루프 유지, Swin은 logits만 반환 |
| [explanation_plans_files.md](../../../documentation/explanation_plans_files.md) | plans를 통해 전처리·구조·배치 등의 설정 관리 | 새 모델 설정은 별도 이름으로 보관; 기존 plans 덮어쓰기 금지 |
| [pretraining_and_finetuning.md](../../../documentation/pretraining_and_finetuning.md) | 명시적 -pretrained_weights로 로드, segmentation head 제외, 구조/전처리 호환 필요 | 일반 학습에 pretrained encoder가 자동 제공되는 것은 아님 |
| [resenc_presets.md](../../../documentation/resenc_presets.md) | 비교 결과와 ResEnc 예산별 사용법 | Swin 재현용 전체 recipe를 제공하는 문서는 아님 |
| [custom trainer 배포](../../../documentation/how-to/share-models-with-custom-trainers.md) | 추론/재개 환경에서도 trainer import 가능해야 함 | 노트북 외 trainer와 wrapper도 서버에 배포 |

기본 trainer의 `set_deep_supervision_enabled`는 `network.decoder.deep_supervision`에 접근한다. Swin trainer에서는 이를 안전하게 재정의해야 한다. DS OFF는 loss wrapper, target downsampling, 출력 형식에 함께 반영되어야 한다.

현재 U-Net planner의 메모리 추정은 Swin에 대한 추정이 아니다. 완전한 자동 계획을 원한다면 모델별 메모리 추정과 planner가 별도로 필요하다는 점도 확장 문서에 명시되어 있다.

## 논문에서 확인된 것과 미확인 항목

| 항목 | nnU-Net 기반 SwinUNETR V2 |
|---|---|
| 실행 프레임워크 | nnU-Net에 통합 |
| pretrained encoder | 없음, from scratch |
| 분할 | 모델 간 공통 nnU-Net 분할, 5-fold |
| GPU 상한 | 단일 A100 40GB |
| ACDC Dice / NSD | 92.01% / 95.15%, 5-fold 평균; NSD tolerance 2 mm |
| 정확한 MONAI 버전·Swin trainer commit | 미확인. Auto3DSeg 1.3.0 명시를 Swin 통합 구현 버전으로 옮기면 안 됨 |
| feature_size, depths, heads, window, merging variant | 논문의 해당 실행값 미확인 |
| ACDC patch·batch·spacing | Swin 해당 실행값 미확인 |
| optimizer·LR·weight decay·scheduler | Swin 해당 실행값 미확인 |
| epoch·step·AMP·checkpointing·seed | Swin 해당 실행값 미확인 |
| deep supervision | 본문/보충자료에 적용 여부 설명 없음 |
| best/final 선택·TTA 세부 설정 | 해당 실행값 미확인 |

[본문](https://arxiv.org/html/2404.09556v2), [보충자료](https://papers.miccai.org/miccai-2024/supp/2847_supp.pdf). 보충자료 Table 5의 ACDC 20×256×224, batch 10은 **ResEnc L** 설정이며 Swin 설정이 아니다. MedNeXt L k3의 ACDC 설정은 128³, 1 mm isotropic, batch 2로 제시되어 있다. 모델 간 전처리가 모두 같았다고 해석해서도 안 된다.

공식 저장소 문서의 “All models trained from scratch”라는 요약에는 본문에 명시된 Auto3DSeg 예외가 생략되어 있다. 예외 해석에는 본문을 우선한다. 보충자료에는 별도의 TotalSegmentator 사전학습 STU-Net 실험도 있다.

## MONAI 네트워크 구현의 기본값 — 논문 실험값 아님

확인한 [MONAI 1.3.0 소스](https://raw.githubusercontent.com/Project-MONAI/MONAI/1.3.0/monai/networks/nets/swin_unetr.py) 기준:

| 항목 | 라이브러리 값 |
|---|---|
| feature_size | 24 |
| depths / num_heads | (2,2,2,2) / (3,6,12,24) |
| token patch / window | 2 / 7 |
| MLP ratio / QKV bias | 4 / True |
| dropout / attention dropout / stochastic depth | 0 / 0 / 0 |
| conv normalization / Transformer normalization | InstanceNorm / LayerNorm |
| normalize | True |
| use_checkpoint | False |
| downsample | merging |
| use_v2 | False (V2 사용 시 명시적으로 True) |
| 입력 크기 | 공간축마다 32의 배수 |
| pretrained 자동 로드 | 생성자 자체에서는 수행하지 않음 |

`use_v2=True`는 stage별 residual convolution을 추가한다. `mergingv2`와는 별개의 옵션이다. 1.3.0에는 `img_size` 인자가 있고, [1.5.1](https://raw.githubusercontent.com/Project-MONAI/MONAI/1.5.1/monai/networks/nets/swin_unetr.py)에서는 제거되어 API가 달라졌다. 설치 버전을 고정하고 그 버전에 맞춰 작성해야 한다.

[공식 BTCV 예제](https://raw.githubusercontent.com/Project-MONAI/research-contributions/main/SwinUNETR/BTCV/README.md)는 feature_size 48의 Base와 24의 Small, 12의 Tiny를 구분한다. 라이브러리 기본 24를 사용하면서 논문의 Base를 재현했다고 부르면 안 된다. 이 예제의 96³ 입력 및 SSL 옵션도 Revisited의 ACDC V2 설정을 입증하지 않는다.

## Pretrained encoder 구분

| 대상 | 판정 |
|---|---|
| 우리 baseline 및 네 테마 | 외부 pretrained encoder 없음; PlainConvUNet 새 초기화 |
| Revisited nnU-Net Swin V1/V2 | 없음; 본문 from scratch 명시 |
| Revisited Auto3DSeg Swin | pretrained 사용; 정확한 체크포인트 파일은 미확인 |
| MONAI 공개 SSL 예제 | `model_swinvit.pt`, Swin encoder 자기지도 사전학습 |

공개 SSL 예제는 약 5,050개 3D CT를 사용하고 rotation, inpainting, contrastive 작업으로 학습한다. 이는 ImageNet 2D encoder가 아니다. 이 가중치가 있다는 사실과 현재 실험에서 사용했다는 사실은 다르다. V1 가중치를 V2에 완전히 호환되는 가중치로 간주해서도 안 된다. [공식 pretraining 문서](https://raw.githubusercontent.com/Project-MONAI/research-contributions/main/SwinUNETR/Pretrain/README.md)

## 우리 ACDC 실행 기록 조사

`../acdc-data/runs/`의 baseline 및 네 테마에 저장된 debug.json, plans.json, config.snapshot.json, manifest.json, splits_final.json을 읽었다. 아래 값은 저장 기록 우선, 기록에 없는 내부 상수는 실행 당시 git commit의 코드와 현재 코드가 동일함을 확인한 뒤 보충했다. 전체 추출값과 패키지 목록은 [hyperparameter-audit.json](hyperparameter-audit.json)에 보관한다. 가중치 텐서는 읽지 않았다.

### 데이터와 전처리

| 항목 | 공통 값 |
|---|---|
| 데이터 | Dataset027_ACDC, MRI 1채널, 100명 ED/ES 200개 |
| 출력 | background 0, RV 1, MLV 2, LVC 3 |
| 분할 | fold 0, train 160 / val 40; 환자별 ED/ES 같은 fold |
| 분할 파일 | 다섯 실험의 파일 SHA256 동일 |
| 분할 seed | 기존 ACDC converter의 1234; 일반 nnU-Net 자동 split의 12345와 구분 |
| reader / preprocessor | SimpleITKIO / DefaultPreprocessor |
| 축 순서 | z,y,x; transpose [0,1,2] |
| crop | 비영 영역 bounding box |
| normalization | ZScoreNormalization, use_mask_for_norm=False |
| spacing | [5.0,1.5625,1.5625] mm |
| 전처리 중앙 shape | [18,237,208], 실제 사례마다 다름 |
| 영상 resampling | order 3, order_z 0, force_separate_z=None |
| 정답 resampling | label-aware order 1, order_z 0, force_separate_z=None |
| 예측 확률 복원 | order 1, order_z 0 |
| 구성 | 3d_fullres, cascade 없음 |

### 네트워크

| 항목 | 공통 값 |
|---|---|
| 모델 / encoder | PlainConvUNet / plain convolution encoder |
| stage 수 / 채널 | 6 / [32,64,128,256,320,320] |
| conv / bias | Conv3d / True |
| kernels | 첫 stage [1,3,3], 이후 모두 [3,3,3] |
| strides | [1,1,1], [1,2,2], [2,2,2], [2,2,2], [1,2,2], [1,2,2] |
| conv 횟수 | encoder stage당 2, decoder stage당 2 |
| normalization | InstanceNorm3d, eps=1e-5, affine=True |
| activation | LeakyReLU, inplace=True; negative_slope는 plans에 미명시 (라이브러리 기본값) |
| dropout | 없음 |
| deep supervision | 다섯 실험 모두 ON |
| pretrained | 전달하지 않음; 모델 생성 후 architecture의 initialize 호출 |

### 학습·loss·샘플링

| 항목 | 값 |
|---|---|
| patch / batch | [20,256,224] / 4 |
| epoch | baseline 1000, 각 테마 250 |
| train / val steps per epoch | 250 / 50 |
| 총 optimizer updates | baseline 250,000; 각 테마 62,500 |
| optimizer | SGD; momentum=.99, Nesterov=True, dampening=0 |
| LR / weight decay | .01 / 3e-5 |
| scheduler | .01 × (1−e/E)^.9, e는 0부터, E는 1000 또는 250 |
| warmup / accumulation | 별도 warmup 없음 / accumulation 없음 |
| gradient clipping | L2 norm 상한 12 |
| loss | CE + MemoryEfficientSoftDice, 각각 weight 1 |
| Dice | background 제외, smooth=1e-5, batch_dice=False |
| CE | 배경 포함 4클래스, 별도 클래스 가중치 없음 |
| ignore label | 없음; 증강 padding label −1은 0으로 치환 |
| DS scales | [1,1,1], [1,.5,.5], [.5,.25,.25], [.25,.125,.125], [.25,.0625,.0625] |
| DS weights | 단일 GPU 코드 기준 [8/15,4/15,2/15,1/15,0] |
| foreground oversampling | 설정 .33, probabilistic_oversampling=False |
| 실제 강제 foreground slot | batch4의 마지막 1개; 나머지도 우연히 foreground 포함 가능 |
| train/val 로더 worker | 4 / 2 |
| precision | CUDA autocast + GradScaler |
| torch.compile / DDP | False / False |
| best | EMA foreground pseudo-Dice; EMA=.9×이전+.1×현재 |
| latest 저장 주기 | 50 epochs; final 별도 저장 |
| early stopping | 기본 epoch 예산 종료; 자동 조기중단 없음 |
| 학습 random seed | 실험 전용 고정 seed 확인되지 않음; worker seeds=None |
| cuDNN | deterministic=False, benchmark=True |

DS weighting은 코드의 실제 연산으로 계산했다. “lowest 2 outputs”라는 코드 주석만으로 판단하지 않았다. Dice loss는 구현상 음수가 가능하므로 음수 val loss 자체가 오류는 아니다. 분할 seed 고정은 전체 학습 랜덤성 고정과 다르다.

### 증강 상세와 실험별 활성화

baseline은 아래의 전체 기본 증강을 사용한다. 각 확률은 매 사례/패치에 대한 바깥 적용 확률이고, 채널 확률은 별도이다.

| 증강 | 범위·확률 |
|---|---|
| 공간 처리 | dummy 2D=True: 깊이를 묶고 평면 회전/확대축소 후 3D 복원 |
| rotation | p=.2, −π~π |
| scaling | p=.2, .7~1.4; 공간축 간 배율 동기화 |
| elastic | p=0 |
| spatial crop | random_crop=False; 로더에서 이미 무작위 패치 샘플링 |
| image/seg 보간 | bilinear; image padding 0, seg padding −1 |
| Gaussian noise | p=.1, noise_variance 인자 (0,.1), channel p=1 |
| Gaussian blur | p=.2, sigma .5~1 voxel, channel p=.5 |
| brightness multiplier | p=.15, .75~1.25, channel p=1 |
| contrast | p=.15, .75~1.25, preserve_range=True |
| low resolution | p=.25, scale .5~1, channel p=.5, ignore_axes=(0,) |
| inverted gamma | p=.1, gamma .7~1.5, invert=True, retain_stats=True |
| ordinary gamma | p=.3, gamma .7~1.5, invert=False, retain_stats=True |
| mirror | allowed_axes=(0,1,2) |

noise_variance는 API 인자명 그대로 기록했다. sigma인지 variance인지, mirror의 내부 확률 등 외부 패키지 내부 동작은 해당 실행 버전 소스까지 확인해야 하므로 임의로 추가 확정하지 않는다. 단일 채널에서 blur의 전체 적용 확률은 .2×.5=.1, low resolution은 .25×.5=.125이다.

| 실험 | epochs | 활성 증강 | gamma | DS |
|---|---:|---|---|---|
| baseline | 1000 | 전체 기본 증강 | 둘 다 ON | ON |
| spatial | 250 | 회전·확대축소·mirror | OFF | ON |
| quality | 250 | noise·blur·low resolution | OFF | ON |
| intensity | 250 | brightness·contrast | OFF | ON |
| none | 250 | 무작위 증강 없음, 샘플링·crop 유지 | OFF | ON |

none에도 샘플링과 padding은 남아 있다. 테마별로 training mirror를 꺼도 inference mirroring은 공통 유지한다.

### 추론·평가와 실행 환경

- sliding-window step=.5, Gaussian blending 사용, mirror axes=(0,1,2), 원본 격자 복원, 4클래스 argmax.
- 정규 학습 종료의 validation 폴더는 final 결과이다. best 비교는 별도 helper가 best 가중치로 재추론하여 보관한다. 패치 pseudo-Dice와 전체 볼륨 Dice/IoU는 다른 지표다.
- 현재 보고서의 NSD tolerance=2 mm. best 비교 helper는 Dice/IoU를 계산하며 NSD는 아직 포함하지 않는다.
- 추가 fold 앙상블이나 connected-component 후처리를 적용하는 실행 경로는 없다.
- GPU는 RTX3090 한 장, Python 3.11.18, torch 2.8.0+cu128, dynamic_network_architectures .4.4, batchgenerators .25.6, numpy 2.4.6, SimpleITK 2.5.7이다.
- **baseline의 batchgeneratorsv2는 .3.7, 네 테마는 .3.8이다.** 동일 환경에서 증강만 바뀌었다고 엄밀하게 표현할 수는 없다. 이 차이가 성능 차이의 원인이라는 증거도 아직 없다.
- baseline commit: `22466dd87b2c4af4f9e9af4b9ff1e874dd70adfc`; 테마 commit: `92ed9fad32eca561ad133592b94230fe953ecf4e`.
- 해당 두 commit의 기본 trainer·PolyLR·network factory를 현재 파일과 비교해 동일함을 확인했다. 실행 당시 manifest에는 수정 파일 기록도 있으므로 git hash 하나만으로 모든 실행 파일이 불변이었다고 주장하지 않는다.

## 다음 Swin 비교에 적용할 판단

1. nnU-Net trainer 기반, pretrained 없음, DS OFF는 현재 목표와 맞는다.
2. 같은 조건 대조군으로 U-Net DS OFF / 250 epochs / 기본 전체 증강을 새로 둔다. 기존 다섯 결과는 DS ON이므로 직접적인 구조 단독 대조군이 아니다.
3. SGD .01을 유지하면 “같은 nnU-Net 학습 recipe에서의 구조 비교”이다. 논문 당시 Swin optimizer의 재현이라고 부르지 않는다.
4. feature_size 24/48, patch padding 방식, MONAI 버전, gradient checkpointing, batch는 아직 새 실험을 위해 선택·검증할 값이다. 라이브러리 기본값과 논문값을 혼동하지 않는다.
5. [20,256,224]를 [32,256,224]로 내부 padding 후 출력 crop하는 방식은 제안일 뿐 논문에서 확인된 방식이 아니다. 메모리는 서버에서 실제 backward까지 확인한다.
6. 두 새 실험의 패키지 버전·분할·실행 예산·평가 조건을 고정하고 VRAM 및 시간을 함께 기록한다. 250 epoch는 동일 update 예산이지 동일 FLOPs나 동일 시간 예산이 아니다.

훈련 코드·기존 노트북·데이터·가중치는 이번 조사에서 변경하지 않았다.
