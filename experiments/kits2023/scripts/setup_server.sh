#!/usr/bin/env bash
# Run only on the Linux NVIDIA server. All environments/caches live outside Git.
set -euo pipefail
if [[ "$(uname -s)" != Linux ]] || ! command -v nvidia-smi >/dev/null; then
  echo 'This installer requires a Linux NVIDIA GPU server. No changes made.' >&2
  exit 1
fi
if [[ $# -ne 1 || "$1" != /* ]]; then
  echo 'Usage: bash experiments/kits2023/scripts/setup_server.sh /absolute/storage/root' >&2
  exit 1
fi
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd -- "$recipe_dir/../.." && pwd)"
storage_dir="$1"
case "$storage_dir/" in "$repo_dir/"*) echo 'Storage must be outside the checkout' >&2; exit 1;; esac
# Python 3.11 is the supported bootstrap environment; install it via your OS first.
python3.11 -m venv "$storage_dir/venv"
export PIP_CACHE_DIR="$storage_dir/cache/pip"
export XDG_CACHE_HOME="$storage_dir/cache"
"$storage_dir/venv/bin/python" -m pip install --upgrade pip
# Explicit, reproducible CUDA wheel selection; the NVIDIA driver must support CUDA 12.8.
"$storage_dir/venv/bin/python" -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
"$storage_dir/venv/bin/python" -m pip install 'torch==2.8.0' -e "$repo_dir" -r "$recipe_dir/requirements-server.txt"
"$storage_dir/venv/bin/python" -m pip check
"$storage_dir/venv/bin/python" - "$recipe_dir" "$storage_dir" "$repo_dir" <<'PY'
import json, pathlib, subprocess, sys
recipe, root, repo = map(pathlib.Path, sys.argv[1:])
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable: check NVIDIA driver/CUDA wheel compatibility before continuing')
cfg_path = recipe/'config.server.json'
if not cfg_path.exists():
    c = json.loads((recipe/'config.json').read_text())
    c.update(storage_root=str(root), source_dataset=str(root/'source'/'kits23'/'dataset'),
             execution_enabled=True,
             expected_git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip())
    cfg_path.write_text(json.dumps(c, indent=2) + '\n')
(root/'environment.lock.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
print('Server config:', cfg_path)
print('GPU:', torch.cuda.get_device_name(0))
PY
printf '\nNext: source "%s/venv/bin/activate"\n' "$storage_dir"
printf 'Then: cd "%s" && jupyter lab\n' "$repo_dir"
echo 'Open experiments/kits2023/notebooks/01_prepare.ipynb'
