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

## 로컬 검증

표준 라이브러리 기반 검사(네트워크·GPU 없음):

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s experiments/acdc/tests -v
```

numpy/nibabel/pandas/surface-distance가 있으면 합성 NIfTI 평가 테스트도 실행합니다. 없으면 해당 테스트만 건너뜁니다. 실제 다운로드·전처리·CUDA 학습은 GPU 서버에서 검증해야 합니다.
