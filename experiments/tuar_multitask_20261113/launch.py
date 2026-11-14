"""Portable queue for the frozen nine-run experiment; no training on import."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def joint(mode):
    return ('joint_' + mode + '_seed42', mode, ['mi', 'ssvep_2s', 'tuar'])


QUEUES = {
    '3': [('independent_mi_seed42', 'plain', ['mi']), joint('plain'), joint('homo_fixed')],
    '4': [('independent_tuar_seed42', 'plain', ['tuar']), joint('adapter'), joint('hetero_fixed')],
    '6': [('independent_ssvep_2s_seed42', 'plain', ['ssvep_2s']), joint('homo_gate'), joint('hetero_gate')],
}


def commands(queue):
    runs = [r for q in QUEUES.values() for r in q] if queue == 'all' else QUEUES[queue]
    return [[sys.executable, '-u', str(ROOT / 'train.py'), '--name', name,
             '--mode', mode, '--tasks', *tasks, '--steps', '1500', '--batch', '16',
             '--eval-every', '100', '--seed', '42'] for name, mode, tasks in runs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', choices=[*QUEUES, 'all'], required=True,
                        help='Historical queue label; independent of device ID.')
    parser.add_argument('--device', required=True,
                        help='One device ID or UUID passed to CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    cmds = commands(args.queue)
    if args.dry_run:
        print(json.dumps({'CUDA_VISIBLE_DEVICES': args.device, 'commands': cmds}, indent=2))
        return
    # Linux flock is released automatically on exit. Different queue labels must
    # also serialize if they target the same device. All queues share run locks.
    try:
        import fcntl
    except ImportError as exc:
        raise SystemExit('The training launcher requires Linux; --dry-run works on Windows.') from exc
    import hashlib
    lock_name = hashlib.sha256(args.device.encode()).hexdigest()[:16]
    device_lock = (ROOT / ('queue_device_' + lock_name + '.lock')).open('a')
    fcntl.flock(device_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.device,
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    (ROOT / 'logs').mkdir(exist_ok=True)
    for cmd in cmds:
        name = cmd[cmd.index('--name') + 1]
        run_lock = (ROOT / ('queue_' + name + '.lock')).open('a')
        try:
            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (ROOT / 'results' / name / 'complete.json').exists():
                print('ALREADY_COMPLETE', name, flush=True)
                continue
            tasks = cmd[cmd.index('--tasks') + 1:cmd.index('--steps')]
            for task in tasks:
                if not (ROOT / 'processed' / task / 'manifest.json').is_file():
                    raise FileNotFoundError('Prepare data first: ' + task)
            with (ROOT / 'logs' / (name + '.log')).open('a') as log:
                subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        finally:
            run_lock.close()


if __name__ == '__main__':
    main()
