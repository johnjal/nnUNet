# KiTS 2023 · nnU-Net baseline 재현

**원격 Linux + NVIDIA 서버 전용 실행 구성입니다.** 개발용 Mac에서는 파일을 읽거나 아래 offline smoke test만 실행합니다. 데이터 다운로드, CUDA 설치, 전처리, 학습은 서버에서만 합니다. 기존 nnU-Net 모델/학습 코드는 변경하지 않습니다.

목표: `nnU-Net Revisited` (arXiv:2404.09556v2)의 KiTS2023 `nnU-Net (org.)` 행. 489 cases, 5-fold, Dice 86.04%, NSD(2mm) 82.91%. 기본 `nnUNetTrainer / nnUNetPlans / 3d_fullres`, 기본 8GB 계획 목표를 사용합니다. 단일 RTX 3090에서 fold를 순차 실행합니다.

## 새 서버에서 시작

필요 조건: Linux, NVIDIA driver(CUDA 12.8 wheel 지원), Python 3.11 + venv, git, tmux. RAM/디스크 필요량은 전처리와 worker 수에 따라 달라집니다. 데이터 원본, 변환 복사본, 전처리, 체크포인트를 저장할 충분한 외부 디스크를 준비하세요. 노트북에서 실제 여유 공간을 확인합니다.

1. **이 experiments 디렉터리가 포함된 전체 저장소**를 서버로 복사하거나 해당 변경을 포함한 Git 커밋을 checkout합니다. 공식 upstream만 새로 clone하면 이 실험 파일들은 포함되지 않습니다.
2. 서버에서 저장 경로를 하나 지정해 환경을 만듭니다.

```bash
bash experiments/kits2023/scripts/setup_server.sh /data/kits2023-reproduction
source /data/kits2023-reproduction/venv/bin/activate
jupyter lab
```

설치 스크립트는 서버 CUDA를 검사하고 `config.server.json`을 생성합니다. 해당 설정만 실행을 허용합니다. 기본 `config.json`은 항상 비활성화 상태로 배포합니다. 학습이 자동으로 시작되지는 않습니다. 기존 서버 설정은 덮어쓰지 않습니다. Python/드라이버 설치는 OS 관리자 환경에 맞게 먼저 진행하세요.

3. 다음 노트북을 순서대로 실행합니다.

| 노트북 | 역할 |
|---|---|
| `notebooks/01_prepare.ipynb` | 서버 점검, 공식 데이터 다운로드, 변환, 전처리, 5-fold 생성 |
| `notebooks/02_train_monitor.ipynb` | tmux 학습 시작/재개, 동일 그래프의 train/validation loss, 상태·GPU 메모리·로그 확인 |
| `notebooks/03_evaluate.ipynb` | 전체 볼륨 예측의 영역별 Dice/NSD, CSV·JSON 보고서 |

01번과 02번의 장시간 작업은 tmux에서 실행되므로 Jupyter 브라우저/커널 종료에도 계속됩니다. 준비 완료 후 학습을 시작하세요. 셀을 반복 실행해도 이미 실행 중인 세션을 중복 생성하지 않습니다.

기본은 `folds: [0]`입니다. fold 0을 평가한 후 `config.server.json`에서 `folds: [0,1,2,3,4]`로 수정하고 02번의 설정 셀과 시작 셀을 다시 실행합니다. 완료 fold는 건너뜁니다. 03번의 설정 셀도 다시 실행해야 전체 fold를 평가합니다. 전체 5-fold가 완료되기 전 결과는 부분 결과입니다.

## 데이터 다운로드와 재시작

[KiTS23 공식 저장소](https://github.com/neheller/kits23)의 학습 케이스 목록(0–299, 400–588)을 사용합니다. 시험 케이스는 다운로드하지 않습니다. 영상은 공식 다운로드 코드가 지정한 `neheller/KiTS-Challenge-Imaging`에서, 라벨은 KiTS23 GitHub에서 받습니다. 현재 공식 README의 버전은 0.1.4이며 0.1.3 이후 변경은 의존성 업데이트라고 설명합니다.

- Annotation commit: `c1088353084c17b8882a11db71429e7c022b7785`로 고정.
- 영상 Hugging Face revision은 첫 실행 때 SHA로 고정하고 `source/kits23/dataset/download_manifest.json`에 저장.
- 완료 파일 SHA256 및 gzip CRC를 검증합니다. 중단하면 검증된 파일을 건너뛰고 미완료 파일부터 다시 받습니다. 개별 중단 파일은 처음부터 다시 받습니다.
- **다른 서버에서 동일 데이터를 재현하려면** 이 manifest를 새 서버의 `source_dataset/download_manifest.json`에 먼저 복사하세요. 그러면 같은 영상 revision과 체크섬을 사용합니다.
- nnU-Net의 공식 `Dataset220_KiTS2023` 변환기를 사용합니다. 다운로드 manifest도 실험 폴더에 보관합니다.
- 데이터는 CC BY-NC-SA입니다. [공식 라이선스 안내](https://github.com/neheller/kits23#license-and-attribution)를 확인하세요.

## 저장 위치와 학습 관리

모든 대용량 파일과 캐시는 `storage_root` 아래에 생성합니다.

```text
storage_root/
  venv/                         # 서버 전용 환경
  cache/                        # pip, matplotlib 등
  source/kits23/dataset/         # 공식 다운로드 + manifest
  nnUNet_raw/Dataset220_KiTS2023/
  nnUNet_preprocessed/Dataset220_KiTS2023/
  runs/<run_id>/
    manifest.json               # Git, Python, pip freeze, GPU 정보
    config.snapshot.json        # 학습 설정과 입력 metadata 해시
    splits_final.json           # 고정된 train/validation 목록
    nnUNetPlans.json
    download_manifest.json
    logs/                       # console + 15초 간격 GPU 사용량 샘플
    fold_0.status.json          # 실행/완료/실패, PID, 시간
    nnUNet_results/             # nnU-Net 체크포인트, 로그, validation 예측
    reports/                    # 케이스/영역/fold 결과 CSV + 비교 JSON
```

같은 storage_root의 runner는 GPU별 파일 잠금으로 중복 실행을 막습니다. 다른 storage_root 또는 수동 실행 프로세스까지 관리하지는 않습니다. 준비 과정은 별도 잠금을 사용합니다. 여러 환경이 하나의 전처리 경로를 동시에 수정하지 않도록 별도의 storage_root를 사용하세요.

학습 중단 뒤 `RESUME=True`로 시작하면 `--c`로 재개합니다. nnU-Net은 주기적 체크포인트를 저장하므로 마지막 저장 이후의 epoch는 재실행될 수 있습니다. 로그 parser는 재개 후 같은 epoch를 최신 값으로 대체합니다. 체크포인트 없이 실패한 폴더는 로그를 확인한 뒤 서버에서 별도 이름으로 옮기고 다시 시작하세요. 자동 삭제하지 않습니다. 학습 조건이나 plans/splits 변경 시 같은 run_id 재사용을 거부합니다.

기본 trainer는 1,000 epochs × 250 training iterations, epoch당 50 validation iterations입니다. train/validation loss는 패치 기반이며 Dice 항 때문에 음수가 가능합니다. 최종 평가는 전체 검증 영상 예측을 사용합니다. 체크포인트는 이 코드 버전에서 safetensors이며 이전 pth 체크포인트도 관리 스크립트에서 인식합니다.

## 평가 규칙과 재현 한계

- kidney=(1,2,3), masses=(2,3), tumor=(2): 세 개의 겹치는 영역.
- 최종 체크포인트의 validation 출력 사용. 기본 nnU-Net validation의 test-time mirroring 사용. 별도 모델 앙상블/후처리 없음.
- NSD tolerance 2mm, `surface-distance==0.1`. spacing과 affine을 검사하며 평가 중 재샘플링하지 않습니다.
- 케이스별 점수를 영역별로 평균 → fold 내 영역 평균 → fold 동일 가중치 평균. 양쪽 모두 빈 영역은 NaN으로 제외하고, 한쪽만 비면 0점 처리.
- `reports/comparison-folds-*.json`은 전체 5-fold 여부와 재현 조건 차이를 명시합니다.

현재 베이스 코드는 nnU-Net 2.8.1, 작성 시 commit `7bb926124ce58a611fa53d3510c714726c800031`입니다. 서버 설치 시 실제 checkout 커밋을 config에 고정하고 실행 때 검사합니다. 학습 코드의 당시 버전, 저자 원본 split 파일, 세부 checkpoint/TTA/집계 조건의 완전한 일치는 확인되지 않았습니다. 따라서 **논문 프로토콜을 따른 현재 코드의 재현 시도**이며 수치의 완전 일치를 보장하지 않습니다. 무작위성 및 3090/A100 차이도 존재합니다. 기본 split은 nnU-Net의 seed 12345 KFold입니다.

서버 bootstrap은 PyTorch 2.8.0 CUDA12.8을 고정하고 나머지 의존성은 설치 후 `environment.lock.txt`와 실험 manifest에 동결 기록합니다. 다른 서버의 정확한 패키지 버전 복원을 위해 첫 서버의 lock을 함께 보관하세요. editable 저장소 경로가 바뀌면 lock의 `-e` 경로를 새 checkout으로 바꾸고 CUDA torch index를 동일하게 지정하세요. 이 lock은 논문 당시 환경 lock이 아닙니다.

공식 논문 코드 링크: https://papers.miccai.org/miccai-2024/562-Paper2847.html

## 로컬 offline smoke test

GPU, PyTorch, 데이터 다운로드 없이 표준 라이브러리만 사용합니다. 테스트 임시 파일은 자동 제거합니다.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s experiments/kits2023/tests -v
```

실제 CUDA 학습, 489건 다운로드, 전체 볼륨 평가는 이 개발 컴퓨터에서 실행하지 않습니다. GPU 서버에서 01번 준비 검증과 첫 fold의 실제 학습/평가로 확인해야 합니다. 로컬에서 만들어진 기존 사용자 파일은 이 구성에서 삭제하지 않습니다.
