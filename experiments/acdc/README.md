# ACDC · nnU-Net baseline 재현

KiTS와 독립된 실험입니다. 기존 KiTS 코드, 데이터, 설정을 변경하거나 삭제하지 않습니다. 실행은 Linux NVIDIA 서버에서만 하며 개발용 Mac에서는 offline smoke test만 수행합니다.

## 왜 1.64GB와 116MB가 다른가?

공식 CREATIS 학습용 폴더의 100명에 대해 파일 크기 메타데이터를 합산했습니다.

| 포함 범위 | 합산 용량 (decimal) |
|---|---:|
| 전체 학습 폴더: 4D cine 영상, 두 시점의 3D 영상/정답 및 보조 파일 | 1,643,086,789 bytes (1.64GB) |
| 이 실험: ED/ES 3D 영상 200개 + 정답 200개 | 115,805,231 bytes (115.8MB, 약 110.4MiB) |

4D는 3D 공간에 시간 축이 추가된 심장 주기 전체 영상입니다. 분할 정답이 제공되는 이완기(ED)·수축기(ES) 두 시점은 별도 3D 파일로 배포됩니다. 이 실험은 해당 파일만 받습니다. 100명을 모두 유지하며 화질 저하, 리사이즈, 환자 서브샘플링은 하지 않습니다. 이는 **압축된 원본 입력**의 크기이며 전처리 결과나 체크포인트 용량이 아닙니다.

공식 데이터: https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html

학습 폴더 메타데이터: https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/folder/63721d7073e9f0047faa0525

## 서버에서 실행

기존 Python 3.11 venv를 Jupyter 커널로 선택합니다. 새 venv나 shell 설치 스크립트는 만들지 않습니다. git, tmux, NVIDIA driver(CUDA12.8 wheel 지원)가 필요합니다.

1. `notebooks/01_prepare.ipynb`를 엽니다. 기본 `STORAGE_ROOT = Path.home() / "acdc-data"`는 root 계정에서 `/root/acdc-data`입니다. **KiTS 저장 경로와 분리하세요.**
2. 0-1 서버 확인 → 0-2 패키지 설치 → 0-3 GPU 확인/서버 설정 생성을 실행합니다. 기존 nnU-Net venv의 필요한 패키지가 이미 설치됐다면 0-2를 생략할 수 있습니다.
3. 최초 설치 후 **커널을 재시작**하고 `1. 실험 설정 불러오기`부터 진행합니다. editable 설치 직후에는 기존 커널에 경로가 적용되지 않을 수 있습니다.
4. 01번 다운로드/전처리 시작 셀을 실행합니다. tmux의 `acdc-prepare` 세션에서 실행됩니다. 준비 상태 셀이 `완료`를 표시하면 다음으로 이동합니다.
5. `02_train_monitor.ipynb`에서 fold 0을 시작하고 train/validation loss를 같은 그래프에서 확인합니다. 브라우저가 끊겨도 tmux 학습은 계속됩니다.
6. `03_evaluate.ipynb`에서 전체 볼륨 검증 예측의 Dice/NSD를 평가합니다.
7. 전체 실험은 `config.server.json`의 `folds`를 `[0,1,2,3,4]`로 변경한 후 02번 설정 셀부터 다시 실행합니다. 완료 fold는 건너뜁니다. 중단된 fold를 이어갈 때는 `RESUME=True`로 설정합니다. 평가 노트북의 설정도 다시 로드하세요.

기존 서버 설정이 있으면 저장된 storage_root를 우선 사용합니다. 경로를 바꾸려면 config.server.json의 storage_root 및 source_dataset을 함께 바꾸고, 실행 중인 작업과 기존 결과 이동 여부를 먼저 확인하세요. 단순 경로 변경이 기존 데이터를 이동시키지는 않습니다.

다른 실험의 `workflow` 모듈을 이미 불러온 커널에서 ACDC를 열면 명시적으로 중단합니다. 커널을 재시작해서 서로 다른 실험의 설정이 섞이지 않게 하세요.

## 다운로드 및 저장 공간

`data_manifest.json`은 공식 파일 400개의 이름, item ID, URL, 크기를 고정한 메타데이터입니다. 실제 영상은 저장소에 포함하지 않습니다.

- 최대 **10개 파일을 동시 다운로드**합니다. 기존 config.server.json에 download_workers가 없어도 기본값 10을 사용합니다. 필요하면 1–10으로 낮출 수 있습니다.
- 완료 파일을 재사용하며, 실패한 파일이 있어도 다른 성공 파일의 기록은 보존합니다. manifest는 메인 스레드에서만 저장합니다.
- 4D 영상, 시험 환자 50명, zip 아카이브를 다운로드하지 않습니다.
- 공식 프레임을 `nnUNet_raw/Dataset027_ACDC/imagesTr` 또는 `labelsTr`에 바로 다운로드합니다. 원본을 별도로 복사하지 않습니다.
- 파일은 `.partial`로 받은 뒤 크기와 gzip CRC를 확인하고 최종 이름으로 교체합니다.
- 첫 다운로드의 SHA256을 서버 manifest에 저장합니다. 재실행은 검증된 파일을 재사용합니다. 공식 API에 체크섬이 없어 최초 SHA256은 최초 다운로드 시 확정합니다.
- `source_dataset` 설정은 이름을 호환성상 유지하지만 실제로는 **영상이 없는 metadata 디렉터리**입니다.
- pip 다운로드 캐시는 설치 셀에서 비활성화합니다. 다른 캐시는 storage_root/cache에 둡니다. 이미 존재하는 기존 캐시는 자동 삭제하지 않습니다.
- 다운로드·전처리·학습 시작 전 기본 5GiB 여유 공간을 검사합니다. 이는 최소 시작 조건이며 작업 중 공간 사용량에 대한 보장은 아닙니다. 50GB 총량 중 venv, 기존 KiTS 데이터가 차지하는 공간도 고려하세요.

```text
~/acdc-data/
  metadata/acdc/download_manifest.json
  nnUNet_raw/Dataset027_ACDC/          # 직접 다운로드한 3D 입력 및 마스크
  nnUNet_preprocessed/Dataset027_ACDC/
  cache/
  environment.lock.txt
  runs/acdc-baseline-v1/
    manifest.json                     # 코드/패키지/GPU 환경
    config.snapshot.json              # 설정 및 코드/metadata 해시
    splits_final.json
    download_manifest.json
    logs/                             # 학습 로그, GPU 메모리 샘플
    nnUNet_results/                   # fold별 체크포인트/전체 볼륨 예측
    reports/                          # case, class, fold별 CSV/비교 JSON
```

전처리 완료 후 01번 마지막 셀에서 실제 저장 용량과 여유 공간을 확인할 수 있습니다. checkpoint 보관 및 5-fold 전체 용량은 서버에서 확인합니다. **데이터가 작아도 기본 1,000 epoch 일정은 그대로이므로 학습 시간도 116MB에 비례해 짧아지는 것은 아닙니다.**

## 학습·분할·평가 조건

- `Dataset027_ACDC`, `nnUNetTrainer`, `nnUNetPlans`, `3d_fullres`, 기본 8GB planning target. GPU VRAM을 채우기 위한 별도 확대 없음.
- 기본 1,000 epochs × 250 training iterations, epoch당 50 validation iterations.
- 채널 `cineMRI`; background=0, RV(우심실)=1, MLV(심근)=2, LVC(좌심실)=3. KiTS의 겹치는 영역 평가를 사용하지 않습니다.
- 저장소 공식 변환기의 `create_ACDC_split(..., seed=1234)` 사용. 환자 단위 5-fold, fold당 train 80명/160영상 및 validation 20명/40영상. 같은 환자의 ED/ES가 서로 다른 집합에 들어가는 것을 검증합니다.
- 최종 checkpoint의 validation 예측 사용. 기본 test-time mirroring 유지. 별도 앙상블·후처리 없음.
- NSD tolerance 2mm. 케이스 평균 → 클래스 평균 → fold 평균. 둘 다 빈 마스크는 NaN 제외, 한쪽만 비면 0점. geometry가 다르면 임의 resampling 없이 오류 처리.
- 논문 비교값: `nnU-Net (org.)` ACDC Dice 91.54%, NSD 94.93% (arXiv:2404.09556v2 Tables 1, 4).

이 구성은 현재 코드로 논문 프로토콜을 재현하려는 실험입니다. 저자의 원본 split, 당시 코드·패키지, checkpoint/TTA 및 세부 집계 방식의 완전한 일치는 아직 확인되지 않았습니다. 그 차이는 manifest와 보고서에 남깁니다. 기본 설정은 fold 0만 실행하며 5-fold 완료 전에는 논문의 전체 평균과 동등한 결과로 취급하지 않습니다.

설정 파일은 실행 비활성화 상태로 배포합니다. 서버에서 설치한 실제 Git 커밋과 패키지 목록을 기록합니다. 같은 run_id에서는 설정/코드/패키지/plans/splits가 변경되면 재사용을 거부하며 fold 목록 확장만 허용합니다.

## 저장된 실험 결과

[docs/README.md](docs/README.md)에 fold 0의 1,000 epoch 완료 결과, 평가표, 학습/GPU 차트와 예측 예시를 정리했습니다. 원본 영상·예측 볼륨·체크포인트 없이 읽을 수 있습니다.

## 전체 데이터 loss 분석 (새 서버에서도 실행 가능)

[04_loss_analysis.ipynb](notebooks/04_loss_analysis.ipynb)는 저장소 옆에 복사한 `acdc-data`를 자동으로 찾습니다. 초기화·현재 venv 패키지 설치 셀을 포함하며 기존 서버의 절대 경로 설정에 의존하지 않습니다. 새 서버의 venv 커널에서 `INSTALL_PACKAGES=True`로 설치한 뒤 커널을 재시작하고, `RUN_INFERENCE=True`로 설정하면 저장된 fold 0 모델로 train 160개와 validation 40개의 확률/마스크를 생성합니다. 기존 결과는 유지하며, 완료한 사례는 재실행 시 건너뜁니다. 추론은 Linux/CUDA 서버에서만 허용합니다.

이후 전체 볼륨 평가 loss(`CE − foreground soft Dice`)의 histogram, train/validation 비교, 클래스별 오류, 높은 loss 사례 목록을 확인할 수 있습니다. 환자 번호와 ED/ES frame을 선택하면 슬라이더 또는 슬라이스 번호로 MRI·정답·예측·오류·CE 지도를 탐색합니다. 이 값은 학습 당시 패치/deep-supervision loss와 다릅니다.

새 대용량 결과는 `acdc-data/runs/<run_id>/loss_analysis/`에만 저장됩니다. 확률 출력이 있는 경우 분석/시각화는 CPU에서도 가능합니다. 원본 raw와 모델·분할 파일이 필요하며, 원본 폴더를 수정하거나 다시 다운로드하지 않습니다.

## 학습 epoch 선택 근거

[05_epoch_selection.ipynb](notebooks/05_epoch_selection.ipynb)는 빈 로컬 venv에서 시작할 수 있도록 커널 등록 안내와 분석 패키지 설치 셀을 포함합니다. 보관된 학습 로그만 읽으며 GPU·PyTorch·원본 영상은 필요하지 않습니다. Validation loss 최솟값, 클래스별 Dice에서 변환한 foreground pseudo-mIoU 최댓값/최솟값, 기록 갱신 위치와 경과 epoch를 표시합니다. 50 epoch 평균과 후보 250 epoch의 학습 시간을 함께 비교합니다.

실행하면 `docs/epoch-selection/`에 PNG/PDF 그래프, CSV, 출처 해시가 포함된 JSON과 근거 Markdown을 저장합니다. [생성된 분석 보고서](docs/epoch-selection/README.md)에서도 확인할 수 있습니다. 지표는 패치 기반이며, 기존 best 체크포인트의 EMA 선택 기준과는 구분됩니다.

## 로컬 검증

표준 라이브러리 기반 검사(네트워크·GPU 없음):

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s experiments/acdc/tests -v
```

numpy/nibabel/pandas/surface-distance/SimpleITK가 있으면 합성 NIfTI 평가 테스트도 실행합니다. 없으면 해당 테스트만 건너뜁니다. 실제 다운로드·전처리·CUDA 학습은 GPU 서버에서 검증해야 합니다.

## ACDC 평가의 geometry mismatch 수정

ACDC 원본의 sform과 qform이 달라 nibabel의 affine 비교가 SimpleITK로 내보낸 정상 예측을 거부할 수 있습니다. 평가는 학습·예측과 같은 SimpleITK로 읽고 크기, spacing, origin, direction을 검사합니다. 실제 좌표 불일치는 계속 오류 처리하며 원본 수정이나 resampling을 하지 않습니다. NSD의 spacing도 배열의 z,y,x 순서에 맞춥니다.

이미 학습한 서버에는 `scripts/evaluate_regions.py`와 `scripts/workflow.py`를 함께 반영하고 03번 커널을 재시작한 뒤 처음부터 실행하세요. 재학습이나 재전처리는 필요하지 않습니다. 기존 학습 snapshot은 삭제·갱신하지 않습니다. 평가에 한해 실험 스크립트 해시 변경을 허용하고 학습 당시/평가 당시 해시를 결과 JSON에 기록합니다. 설정, nnUNet 학습 코드, 패키지, plans/splits 및 준비 입력의 검증은 유지합니다. 학습 재개 경로의 엄격한 해시 검증도 그대로입니다. 기존 run 평가 시 학습 당시 config.server.json을 유지하세요.

## 기존 순차 다운로드에서 전환

이미 실행 중인 프로세스는 파일을 수정해도 바뀌지 않습니다. 서버 터미널에서 `tmux attach -t acdc-prepare`로 접속해 Ctrl+C로 기존 준비 작업을 중단하고 종료를 확인하세요. 변경 파일을 서버에 반영한 뒤 01번의 다운로드/전처리 시작 셀을 다시 실행하면 완료 파일을 재사용하고 나머지를 10개씩 처리합니다. 서버의 Git 커밋도 업데이트했다면 실행 전 config.server.json의 expected_git_commit을 검토 후 현재 커밋으로 갱신해야 합니다.

## 자동 설정 내부 코드 분석

[자동 결정 항목과 ACDC 적용 결과](docs/automatic-decisions.md)에 데이터 기반 계획, 증강의 조건부/고정 설정, patch sampling, loss, 학습, inference 및 별도 모델 선택 기능을 코드 근거와 함께 정리했습니다.

## 증강 테마 단독 비교

[06_augmentation_themes.ipynb](notebooks/06_augmentation_themes.ipynb)는 fold 0·250 epoch로 `spatial`(회전/mirror/scale), `quality`(noise/blur/low-resolution), `intensity`(brightness/contrast)를 하나씩만 켜서 비교합니다. 기존 전처리와 기본 학습 설정을 공유하고 Gamma/elastic은 모두 끕니다. 선택한 증강의 확률·범위는 기본 trainer 값을 유지합니다. spatial 외 테마는 공간 변환 확률을 0으로 두되 최종 patch crop 처리는 유지합니다.

서버에서 초기 설정 후 `START_SUITE=True`로 실행하면 tmux가 세 테마와 무증강 대조군, 총 네 실험을 순차 진행합니다. 중단 시 `RESUME=True`로 다시 시작할 수 있습니다. 각 결과는 `acdc-data/runs/<SUITE_ID>-<theme>/`에 저장되고, 전체 상태·비교 CSV/JSON은 `acdc-data/augmentation_suites/<SUITE_ID>/`에 저장됩니다. 노트북에서 train/validation loss와 최종 전체 볼륨 Dice·mIoU를 확인합니다. `INCLUDE_NO_AUG_CONTROL=True`가 기본값이므로 증강 없는 대조군도 함께 실행합니다.

각 실험은 새 무작위 초기화이며 단일 seed를 맞추는 변경은 하지 않습니다. 최종 평가는 모든 테마에서 공통 default mirroring TTA를 유지합니다. 250 epoch 일정은 기존 1,000 epoch의 앞부분과 LR 일정이 다르며, 단일 fold의 결과만으로 통계적 효과를 확정하지 않습니다. 모델별 실제 GPU 학습·최종 평가는 서버에서 실행해야 합니다.

구현: [전용 trainer](../../nnunetv2/training/nnUNetTrainer/variants/data_augmentation/nnUNetTrainerACDCTheme250.py), [순차 큐](scripts/run_augmentation_suite.py). 기본 trainer와 기존 baseline 실행 설정은 변경하지 않았습니다.
