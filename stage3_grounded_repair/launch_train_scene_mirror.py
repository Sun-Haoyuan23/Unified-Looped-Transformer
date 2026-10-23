"""Finite late-check Linear FM rebridge probe; do not interrupt other jobs."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time
from launch_grounding_diagnostic import GPUS, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--experiment', required=True)
    p.add_argument('--teacher-diagnostic', type=Path, required=True)
    p.add_argument('--gpus', default='0,4,6,7')
    p.add_argument('--identity-audit',type=Path,required=True)
    args = p.parse_args()
    gpu_ids = [int(x) for x in args.gpus.split(',')]
    if len(gpu_ids) != len(set(gpu_ids)) or any(g not in GPUS for g in gpu_ids):
        raise ValueError('Invalid healthy GPU selection')
    root = args.work / 'stage3_grounded_repair'
    out = root / args.experiment
    out.mkdir(parents=True, exist_ok=True)
    def worker(shard, gpu):
        path = out / f'worker_{shard}.json'
        status = {'shard': shard, 'gpu_physical': gpu, 'gpu_uuid': GPUS[gpu],
                  'status': 'waiting_for_gpu', 'started': time.time()}
        write(path, status)
        try:
            deadline = time.time() + 1800
            while True:
                q = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
                if int(q.stdout.strip()) < (5000 if gpu == 1 else 2000):
                    break
                if time.time() > deadline:
                    raise TimeoutError('Existing workload occupied GPU; no process killed')
                time.sleep(20)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       TORCHDYNAMO_DISABLE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONWARNINGS='ignore', PYTHONPATH=str(args.work / 'deps'))
            cmd = ['/home/shy_23/venvs/show_o2/bin/python', str(root / 'build_train_scene_mirror_candidates.py'),
                   '--work', str(args.work), '--states', str(root / 'states/train_pilot_minimal_v14'),
                   '--teacher-diagnostic', str(args.teacher_diagnostic), '--identity-audit',str(args.identity_audit), '--output', str(out / f'shard{shard}'),
                   '--shard', str(shard), '--num-shards', str(len(gpu_ids))]
            with (out / f'candidates_shard{shard}.log').open('w') as log:
                proc = subprocess.Popen(cmd, env=env, cwd=args.work, stdout=log, stderr=subprocess.STDOUT)
                status.update(status='candidates', pid=proc.pid)
                write(path, status)
                code = proc.wait(timeout=1800)
                if code:
                    raise RuntimeError(f'Candidate worker exited {code}')
            status.update(status='generation_complete', completed=time.time())
        except Exception as error:
            status.update(status='failed', error=str(error), completed=time.time())
        write(path, status)
        return status
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        result = list(pool.map(lambda item: worker(*item), enumerate(gpu_ids)))
    success = all(r['status'] == 'generation_complete' for r in result)
    write(out / 'generation_status.json', {'success': success, 'workers': result,
                                         'expected_images': 288, 'optimizer_steps': 0})
    if not success:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
