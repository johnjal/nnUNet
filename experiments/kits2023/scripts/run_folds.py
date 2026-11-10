"""Run a sequential fold queue in tmux; locks prevent concurrent writers."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time
from workflow import (freeze_run, guard, model_dir, paths, read_config,
                      training_command, environment, write_json)


def action_for(folder, resume):
    folder = Path(folder)
    if (folder/'validation'/'summary.json').is_file() and any(folder.glob('checkpoint_final.*')):
        return 'skip'
    if folder.exists() and any(folder.iterdir()):
        if not resume:
            raise RuntimeError(f'Existing output: {folder}. Use --resume, or choose a new run_id.')
        if not any(folder.glob('checkpoint_*.pth')) and not any(folder.glob('checkpoint_*.safetensors')):
            raise RuntimeError(f'No resumable checkpoint in {folder}; inspect before moving failed output aside.')
        return 'resume'
    return 'start'


def run(c, resume=False):
    guard(c)
    p = paths(c)
    p['root'].mkdir(parents=True, exist_ok=True)
    # Per-GPU lock also prevents different run_ids from exhausting the same GPU.
    lock = p['root']/f'gpu-{c["gpu"]}.lock'
    with lock.open('w') as handle, (p['root']/f'run-{c["run_id"]}.lock').open('w') as run_lock:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another runner owns this run or GPU {c["gpu"]}') from None
        freeze_run(c)
        (p['run']/'logs').mkdir(exist_ok=True)
        for fold in c['folds']:
            folder = model_dir(c)/f'fold_{fold}'
            action = action_for(folder, resume)
            if action == 'skip':
                print(f'Fold {fold}: already complete', flush=True)
                continue
            command = training_command(c, fold, action == 'resume')
            state = {'fold': fold, 'command': command, 'status': 'running', 'started': time.time()}
            status_path = p['run']/f'fold_{fold}.status.json'
            write_json(status_path, state)
            try:
                with (p['run']/'logs'/f'fold_{fold}.console.log').open('a', buffering=1) as log:
                    log.write('\n' + json.dumps(state) + '\n')
                    process = subprocess.Popen(command, env=environment(c), stdout=log, stderr=subprocess.STDOUT)
                    state['pid'] = process.pid
                    write_json(status_path, state)
                    while True:
                        try:
                            code = process.wait(timeout=15)
                            break
                        except subprocess.TimeoutExpired:
                            # Device-wide memory includes any other processes; lock covers our runners only.
                            sample = subprocess.run([
                                'nvidia-smi', '-i', c['gpu'],
                                '--query-gpu=timestamp,memory.used,utilization.gpu', '--format=csv,noheader,nounits'
                            ], capture_output=True, text=True)
                            if sample.returncode == 0:
                                with (p['run']/'logs'/f'fold_{fold}.gpu.csv').open('a') as gpu_log:
                                    gpu_log.write(sample.stdout)

                if code:
                    raise RuntimeError(f'Fold {fold} failed (exit {code}); see console log')
                if action_for(folder, True) != 'skip':
                    raise RuntimeError('Process exited without final checkpoint and validation summary')
                state['status'] = 'completed'
            except BaseException as error:
                # A keyboard interrupt terminates the child so it cannot outlive its lock.
                if 'process' in locals() and process.poll() is None:
                    process.terminate()
                    process.wait()
                state.update(status='failed', error=str(error))
                raise
            finally:
                state['ended'] = time.time()
                write_json(status_path, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(read_config(args.config), args.resume)
