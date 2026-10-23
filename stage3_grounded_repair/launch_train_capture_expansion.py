"""Freeze 24 TRAIN prompts x2 seeds, capture index35, and verify zero replay.

A development repair only authorizes checking an interface on new train data;
it is never selected as a training target. No optimizer update is performed.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from candidate_provenance import file_hash,freeze_config
from prepare_train_expansion import prepare
from launch_grounding_diagnostic import GPUS,write

p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True)
p.add_argument('--gpus',default='0,4,6,7');p.add_argument('--count',type=int,default=32);args=p.parse_args()
root=args.work/'stage3_grounded_repair';out=root/'train_pilot_expansion_v24'
states=root/'states/train_pilot_expansion_v24';out.mkdir(exist_ok=True)
tasks=out/'train_prompts.json';chosen=prepare(root/'splits_v1/train.jsonl',root/'train_pilot_minimal_v14/train_prompts.json',tasks,count=args.count,seeds=(101,202))
if len(chosen)!=args.count or any(r['split']!='train' for r in chosen):raise ValueError('Expansion must contain exactly the requested TRAIN count')
gpus=[int(s) for s in args.gpus.split(',')]
if len(set(gpus))!=len(gpus) or any(g not in GPUS or g==1 for g in gpus):raise ValueError('Invalid free GPU selection')
names=['launch_train_capture_expansion.py','prepare_train_expansion.py','test_train_expansion.py','capture_repair_states.py','runtime.py',
    'evidence_encoder.py','evidence_conditioned_revisit.py','candidate_provenance.py']
freeze_config(out/'config.json',dict(tasks_sha256=file_hash(tasks),source_sha256={n:file_hash(root/n) for n in names},
    excluded_tasks_sha256=file_hash(root/'train_pilot_minimal_v14/train_prompts.json'),prompts=args.count,seeds=[101,202],states=2*args.count,indices=[35],split='train',
    gpus=gpus,optimizer_steps=0,development_states_used_for_training=False,
    purpose='Capture new TRAIN baselines and causal states; not yet teacher collection or training'))
snapshot=out/'source';snapshot.mkdir(exist_ok=True)
for name in names:shutil.copy2(root/name,snapshot/name)
def worker(item):
    shard,gpu=item;status=dict(shard=shard,gpu_physical=gpu,gpu_uuid=GPUS[gpu],status='waiting_for_gpu',started=time.time())
    path=out/f'worker_{shard}.json';write(path,status)
    try:
        deadline=time.time()+1800
        while True:
            q=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
            if int(q.stdout.strip())<2000:break
            if time.time()>deadline:raise TimeoutError('GPU occupied; no process interrupted')
            time.sleep(20)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            TORCHDYNAMO_DISABLE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONWARNINGS='ignore',PYTHONPATH=str(args.work/'deps'))
        command=['/home/shy_23/venvs/show_o2/bin/python',str(root/'capture_repair_states.py'),'--work',str(args.work),
            '--tasks',str(tasks),'--output',str(states/f'shard{shard}'),'--shard',str(shard),'--num-shards',str(len(gpus)),
            '--indices','35','--split','train']
        with (out/f'capture_shard{shard}.log').open('w') as log:
            proc=subprocess.Popen(command,env=env,cwd=args.work,stdout=log,stderr=subprocess.STDOUT)
            status.update(status='capture',pid=proc.pid);write(path,status)
            if proc.wait(timeout=2400):raise RuntimeError('Capture failed; inspect shard log')
        status.update(status='capture_complete',completed=time.time())
    except Exception as error:status.update(status='failed',error=str(error),completed=time.time())
    write(path,status);return status
with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
    workers=list(pool.map(worker,enumerate(gpus)))
rows=[json.loads(line) for f in states.glob('shard*/capture_*.jsonl') for line in f.read_text().splitlines()]
success=all(w['status']=='capture_complete' for w in workers)
if success and (len(rows)!=2*args.count or len({r['state_id'] for r in rows})!=2*args.count or any(r['split']!='train' or not r['zero_intervention_test']['passed'] for r in rows)):
    success=False
write(out/'capture_status.json',dict(success=success,workers=workers,captured_states=len(rows),expected_states=2*args.count,
    optimizer_steps=0,train_teachers_qualified=0,next='Train-specific middle geometry and SAME-STATE terminal official/HPS/Pick scoring; do not train from dev'))
if not success:raise SystemExit(1)
