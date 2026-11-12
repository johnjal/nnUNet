"""Read-only source archive and static report. No training, inference or metric recomputation.
Usage: python build_report.py /path/to/acdc-data
Dependencies: numpy, matplotlib, SimpleITK. Output: this script's directory.
"""
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
DOCS = Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS.parent/'scripts'))
from workflow import parse_losses
from evaluate_regions import read_masks


def main(source):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    run = source/'runs/acdc-baseline-v1'
    model = run/'nnUNet_results/Dataset027_ACDC/nnUNetTrainer__nnUNetPlans__3d_fullres'
    fold = model/'fold_0'
    reports = run/'reports'
    figures = DOCS/'figures'
    tables = DOCS/'tables'
    for folder in (figures, tables):
        folder.mkdir(parents=True, exist_ok=True)
    inventory = []
    excluded = []
    # Preserve lightweight originals, including logs and metadata, with source-relative paths.
    candidates = list(run.rglob('*')) + list(source.glob('*'))
    seen = set()
    for file in sorted(candidates):
        if not file.is_file() or file in seen:
            continue
        seen.add(file)
        rel = file.relative_to(source)
        if any(part.startswith('.') for part in rel.parts):
            continue
        if file.suffix.lower() not in {'.json', '.csv', '.log', '.txt', '.png'} or file.stat().st_size > 5*1024**2:
            excluded.append({'source': str(rel), 'bytes': file.stat().st_size})
            continue
        dest = DOCS/'artifacts'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, dest)
        inventory.append({'source': str(rel), 'artifact': str(dest.relative_to(DOCS)),
                          'bytes': file.stat().st_size, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()})
    (DOCS/'artifact_manifest.json').write_text(json.dumps({'copied': inventory, 'excluded_run_files': excluded,
        'excluded_directories': ['nnUNet_raw', 'nnUNet_preprocessed', 'cache', 'metadata'],
        'note': 'Run snapshots retain preparation metadata. Hidden files, locks, checkpoint weights and NIfTI volumes are excluded.'}, indent=2)+'\n')
    def load_csv(path):
        with path.open() as f:
            return list(csv.DictReader(f))
    def save_csv(path, rows):
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    comparison = json.loads((reports/'comparison-folds-0.json').read_text())
    regions = load_csv(reports/'regions-folds-0.csv')
    cases = load_csv(reports/'cases-folds-0.csv')
    assert len(cases) == 120 and len({r['case'] for r in cases}) == 40
    for metric in ('dice', 'nsd'):
        assert np.isclose(np.mean([float(r[metric]) for r in regions])*100, comparison[f'{metric}_percent'])
    epochs = parse_losses(fold)
    assert len(epochs) == 1000 and epochs[-1]['epoch'] == 999
    save_csv(tables/'epoch-history.csv', epochs)
    grouped = {}
    for row in cases:
        grouped.setdefault(row['case'], []).append(row)
    ranked = sorted([{'case': case, 'dice_percent': np.mean([float(r['dice']) for r in rows])*100,
                      'nsd_percent': np.mean([float(r['nsd']) for r in rows])*100}
                     for case, rows in grouped.items()], key=lambda r: (r['dice_percent'], r['case']))
    save_csv(tables/'case-ranking.csv', ranked)
    gpu = np.loadtxt(run/'logs/fold_0.gpu.csv', delimiter=',', usecols=(1, 2))
    status = json.loads((run/'fold_0.status.json').read_text())
    stats = {'epochs': len(epochs), 'runner_hours': (status['ended']-status['started'])/3600,
             'median_epoch_seconds': float(np.median([e['epoch_seconds'] for e in epochs])),
             'gpu_peak_memory_mib': float(gpu[:,0].max()), 'gpu_median_utilization_percent': float(np.median(gpu[:,1])),
             'gpu_samples': len(gpu), 'last_train_loss': epochs[-1]['train_loss'], 'last_val_loss': epochs[-1]['val_loss']}
    (tables/'training-summary.json').write_text(json.dumps(stats, indent=2)+'\n')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(10, 4))
    for key, label in [('train_loss', 'Training'), ('val_loss', 'Validation')]:
        ax.plot([r['epoch']+1 for r in epochs], [r[key] for r in epochs], lw=1, label=label)
    ax.set(xlabel='Epoch (1-based)', ylabel='Loss', title='ACDC fold 0 · training and validation loss')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(figures/'loss-curves.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, label in zip(axes, ['dice','nsd'], ['Dice (%)','NSD @ 2 mm (%)']):
        vals = [float(r[metric])*100 for r in regions]
        bars = ax.bar([r['region'] for r in regions], vals, color=['#36a2eb','#ff9f43','#2ecc71'])
        ax.bar_label(bars, fmt='%.2f'); ax.set(ylim=(0, 105), ylabel=label)
    fig.suptitle('Final whole-volume validation · fold 0 only'); fig.tight_layout()
    fig.savefig(figures/'class-metrics.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(gpu[:,0]/1024, lw=.7); axes[0].set(ylabel='GPU memory (GiB)')
    axes[1].plot(gpu[:,1], lw=.7); axes[1].set(ylabel='Utilization (%)', xlabel='Sample index (approximately every 15 s)')
    fig.suptitle('GPU 0 · device-wide samples (includes any other processes)'); fig.tight_layout()
    fig.savefig(figures/'gpu-usage.png', dpi=150); plt.close(fig)
    examples = []
    for rank, position in [('low', 0), ('middle', len(ranked)//2), ('high', len(ranked)-1)]:
        row = ranked[position]; case = row['case']
        raw = source/'nnUNet_raw/Dataset027_ACDC'
        gt_path = raw/'labelsTr'/f'{case}.nii.gz'
        gt, pred, spacing = read_masks(gt_path, fold/'validation'/f'{case}.nii.gz')
        image, _, _ = read_masks(raw/'imagesTr'/f'{case}_0000.nii.gz', gt_path)
        z = int(np.argmax(np.count_nonzero(gt, axis=(1,2))))
        plane = image[z]; low, high = np.percentile(plane[np.isfinite(plane)], [1,99])
        aspect = spacing[1]/spacing[2]
        fig, axes = plt.subplots(1, 4, figsize=(13, 4))
        for ax, title in zip(axes, ['MRI','Ground truth','Prediction','Disagreement']):
            ax.imshow(plane, cmap='gray', vmin=low, vmax=max(high, low+1), aspect=aspect, interpolation='nearest')
            ax.set_title(title); ax.axis('off')
        colors = ['#2ecc71','#ff9f43','#36a2eb']
        for ax, mask in [(axes[1],gt[z]), (axes[2],pred[z])]:
            ax.imshow(np.ma.masked_equal(mask,0), cmap=ListedColormap(colors),
                      norm=BoundaryNorm([.5,1.5,2.5,3.5],3), alpha=.55, aspect=aspect, interpolation='nearest')
        error = gt[z] != pred[z]
        axes[3].imshow(np.ma.masked_where(~error,np.ones_like(error)), cmap=ListedColormap(['#ff3fcf']),
                       vmin=0,vmax=1,alpha=.75,aspect=aspect,interpolation='nearest')
        fig.suptitle(f"{case} · {rank} score example · slice {z} · whole-volume mean Dice {row['dice_percent']:.2f}%")
        fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(colors+['#ff3fcf'],['RV','MLV','LVC','Mismatch'])],
                   loc='lower center',ncol=4,frameon=False)
        fig.tight_layout(rect=(0,.08,1,.93))
        fig.savefig(figures/f'example-{rank}.png',dpi=130); plt.close(fig)
        examples.append(dict(row, selection=rank, slice_index=z))
    save_csv(tables/'example-selection.csv', examples)
    cfg = json.loads((run/'config.snapshot.json').read_text())['configuration']
    plan = json.loads((run/'nnUNetPlans.json').read_text())['configurations']['3d_fullres']
    region_table = '\n'.join(f"| {r['region']} | {float(r['dice'])*100:.2f} | {float(r['nsd'])*100:.2f} |" for r in regions)
    base = 'artifacts/runs/acdc-baseline-v1'
    text = f'''# ACDC 실험 결과 — acdc-baseline-v1

저장된 서버 결과를 2026-11-12에 정리했습니다. **fold 0만 완료한 결과**이며 5-fold 재현 완료를 뜻하지 않습니다. 학습·추론·지표 재계산 없이 기존 평가 CSV/JSON과 로그를 사용했습니다.

## 평가 결과

| 구분 | Dice (%) | NSD @ 2mm (%) |
|---|---:|---:|
| 이번 실험 — fold 0 | {comparison['dice_percent']:.2f} | {comparison['nsd_percent']:.2f} |
| 기존 보고서의 논문 참고값 — 5-fold 평균 | {comparison['paper_dice_percent']:.2f} | {comparison['paper_nsd_percent']:.2f} |

평가 범위가 달라 위 수치만으로 논문보다 성능이 좋다고 판단할 수 없습니다. 논문의 당시 코드·분할·환경과 완전히 동일함은 확인되지 않았습니다.

| 클래스 | Dice (%) | NSD (%) |
|---|---:|---:|
{region_table}

![클래스별 평가](figures/class-metrics.png)

검증 환자 20명의 ED/ES 영상 40개, 3개 클래스 총 120행입니다. 케이스별 → 클래스별 → fold별 평균이며 양쪽 모두 빈 영역은 제외합니다. 최종 checkpoint의 전체 영상 예측으로 평가했고 NSD tolerance는 2mm입니다. [케이스별 원본 점수]({base}/reports/cases-folds-0.csv), [평균 점수]({base}/reports/regions-folds-0.csv), [평가 조건·해시]({base}/reports/comparison-folds-0.json), [케이스 순위](tables/case-ranking.csv).

## 학습 설정과 완료 상태

- Dataset027_ACDC / nnUNetTrainer / 3d_fullres / nnUNetPlans.
- 환자 단위 5-fold 분할 중 fold 0: 학습 80명·160영상, 검증 20명·40영상.
- **1,000 epoch 완료** (로그 epoch 0~999), epoch당 학습 250회·검증 50회.
- 실제 batch size **{plan['batch_size']}**, patch size `{plan['patch_size']}`, spacing `{plan['spacing']}` mm (계획 배열 축 순서).
- GPU 0: RTX 3090 24GiB, 단일 GPU. GPU 1은 이 실행에 사용하지 않았습니다.
- 학습 실행기 시작~종료: **{stats['runner_hours']:.2f}시간** (최종 validation 포함). epoch 시간 중앙값 **{stats['median_epoch_seconds']:.2f}초**.
- 마지막 Train loss **{stats['last_train_loss']}**, Validation loss **{stats['last_val_loss']}**. 이 loss는 최종 전체 영상 Dice와 다른 지표입니다.
- GPU 샘플 {stats['gpu_samples']:,}개, 장치 전체 메모리 최대 **{stats['gpu_peak_memory_mib']/1024:.2f}GiB**, 사용률 중앙값 **{stats['gpu_median_utilization_percent']:.0f}%**. 프로세스 전용 측정이 아닙니다.
- 학습 시 기록된 Git commit: `{cfg['expected_git_commit']}`. 원본 로그의 시간대는 명시되지 않아 날짜·시각을 임의로 한국 시간으로 바꾸지 않았습니다.

![학습 곡선](figures/loss-curves.png)

[epoch별 정리 CSV](tables/epoch-history.csv) · [시간·GPU 요약](tables/training-summary.json)

![GPU 사용량](figures/gpu-usage.png)

## 예측 예시

전체 영상 클래스 평균 Dice 순위에서 최저, 중간(정렬 인덱스 20), 최고 케이스를 골랐습니다. 각 영상에서 정답 심장 영역이 가장 넓은 z 슬라이스를 표시했습니다. 좋은 사례만 선택하지 않았으며 세 장이 전체 평가를 대신하지 않습니다. 공간 비율을 유지했고 원본 voxel 방향을 사용했습니다.

MRI / 정답 / 예측 / 불일치 영역 순서입니다. RV=초록, MLV=주황, LVC=파랑, 불일치=분홍입니다. 제목 점수는 슬라이스가 아닌 전체 영상 점수입니다. [선택 케이스·슬라이스](tables/example-selection.csv)

![낮은 점수 예시](figures/example-low.png)

![중간 점수 예시](figures/example-middle.png)

![높은 점수 예시](figures/example-high.png)

## 보관한 결과

- `artifacts/runs/acdc-baseline-v1/reports/`: 평가 CSV 3개와 비교 JSON.
- `artifacts/runs/acdc-baseline-v1/nnUNet_results/`: 원본 progress.png, training log, debug 정보, validation summary 및 모델 계획 메타데이터.
- `artifacts/runs/acdc-baseline-v1/logs/`: console log와 GPU 시계열.
- `artifacts/runs/acdc-baseline-v1/`: 완료 상태, 학습 snapshot, 환경 manifest, plans, split, fingerprint, 다운로드 체크섬.
- `artifacts/`의 최상위 로그: 준비·실행 로그와 환경 패키지 목록. 서버 경로는 이력 보존용이며 현재 컴퓨터 경로가 아닙니다.
- `tables/`, `figures/`: 위 자료를 정리한 CSV/JSON과 정적 차트·예시.
- [artifact_manifest.json](artifact_manifest.json): 보관 파일의 원본 상대 경로·크기·SHA-256 및 제외 목록.

**제외:** 원본 MRI·정답 NIfTI, validation 예측 NIfTI, 전처리 배열, checkpoint 가중치, 캐시, 잠금 파일, 숨김/노트북 자동 저장 사본. 데이터는 원래 acdc-data에 그대로 두었습니다. 작은 예시 PNG만 포함했으며 전체 의료 영상 볼륨은 복사하지 않았습니다.

## 다시 생성

`numpy`, `matplotlib`, `SimpleITK`가 설치된 Python에서 `python experiments/acdc/docs/build_report.py /path/to/acdc-data`를 실행합니다. GPU나 PyTorch는 필요 없습니다. 같은 경로의 보고서 파일은 갱신되며 원본 데이터는 읽기만 합니다. 현재 빌더는 이 실행의 fold 0·1,000 epoch·40개 검증 영상에 맞춰 검증합니다. 보고서 열람에는 Python이나 acdc-data가 필요 없습니다.
'''
    pipeline = (DOCS/'training-pipeline.md').read_text()
    text = text.replace('## 평가 결과', pipeline + '\n## 평가 결과', 1)
    (DOCS/'README.md').write_text(text)
    print(json.dumps({'copied_files':len(inventory), 'stats':stats, 'examples':examples},indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]).expanduser().resolve())
