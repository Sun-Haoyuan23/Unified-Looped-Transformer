"""Resume two unstarted shards on free GPUs; stop only own idle launchers."""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import psutil
from launch_grounding_diagnostic import GPUS,write
from candidate_provenance import file_hash

work=Path(sys.argv[1]).resolve();root=work/'stage3_grounded_repair'
experiment='oracle_minimal_teacher_dev12x2_v13';out=root/experiment
launch=json.loads((out/'launch.json').read_text())
if (out/'resource_reschedule.json').exists():raise RuntimeError('Already rescheduled; inspect status')
processes=[]
try:
    for key,script in [('generation','launch_minimal_fm_oracle.py'),('evaluation','evaluate_oracle_pipeline.py')]:
        proc=psutil.Process(launch[key]['pid']);cmd=proc.cmdline()
        if proc.uids().real!=os.getuid() or str(root/script) not in cmd or cmd[cmd.index('--experiment')+1]!=experiment:
            raise RuntimeError('Launcher ownership/identity mismatch')
        proc.suspend();processes.append(proc)
    states={i:json.loads((out/f'worker_{i}.json').read_text()) for i in range(6)}
    if {i for i,s in states.items() if s['status']=='waiting_for_gpu'}!={1,2} or any(states[i]['status']!='generation_complete' for i in (0,3,4,5)):
        raise RuntimeError('Workers are no longer the expected four complete and two unstarted shards')
    if json.loads((out/'evaluation_status.json').read_text())['phase']!='waiting_for_generation':
        raise RuntimeError('Evaluation already running; do not interrupt')
    if any(any('build_minimal_fm_candidates.py' in a or 'score_repair_candidates.py' in a or 'evaluate_images.py' in a for a in child.cmdline()) for proc in processes for child in proc.children(recursive=True)):
        raise RuntimeError('Active model/evaluator child exists; do not interrupt')
    audit=dict(time=time.time(),original_launchers=launch,previous_workers=states,
        new_assignments={'1':0,'2':6},evaluation_gpus=[0,4,6,7],reason='Original GPU2/3 occupied by another user; reuse free GPUs',
        model_prompt_seed_actions_unchanged=True,source_sha256={n:file_hash(root/n) for n in
            ('resume_waiting_minimal_workers.py','evaluate_oracle_pipeline_available.py')})
    write(out/'resource_reschedule.json',audit)
    for proc in processes:proc.kill();proc.wait(timeout=10)
    processes=[]
except Exception:
    for proc in processes:
        try:proc.resume()
        except psutil.NoSuchProcess:pass
    raise
for name in audit['source_sha256']:shutil.copy2(root/name,out/'source'/name)
env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCHDYNAMO_DISABLE='1',
    OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONWARNINGS='ignore',PYTHONPATH=str(work/'deps'))
evaluation=launch['evaluation']['command'].copy();evaluation[1]=str(root/'evaluate_oracle_pipeline_available.py')
with (out/'evaluation_rescheduled.log').open('w') as log:
    p=subprocess.Popen(evaluation,env=env,cwd=work,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
launch['generation_resumed']=dict(pid=os.getpid(),command=sys.argv)
launch['evaluation_resumed']=dict(pid=p.pid,command=evaluation)
write(out/'launch.json',launch)
def worker(item):
    shard,gpu=item;path=out/f'worker_{shard}.json'
    status=dict(shard=shard,gpu_physical=gpu,gpu_uuid=GPUS[gpu],status='waiting_for_gpu',started=time.time(),
        original_assignment=states[shard]['gpu_physical'],resource_only_reschedule=True)
    write(path,status)
    try:
        deadline=time.time()+1800
        while True:
            q=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
            if int(q.stdout.strip())<2000:break
            if time.time()>deadline:raise TimeoutError('Free GPU became occupied; no process killed')
            time.sleep(20)
        worker_env=env.copy();worker_env['CUDA_VISIBLE_DEVICES']=GPUS[gpu]
        command=[sys.executable,str(root/'build_minimal_fm_candidates.py'),'--work',str(work),
            '--states',str(root/'states/oracle_dev12x2_v1'),'--teacher-diagnostic',str(root/'minimal_teacher_geometry_dev6_v12'),
            '--identity-audit',str(root/'minimal_teacher_identity_audit_v13.json'),'--output',str(out/f'shard{shard}'),
            '--shard',str(shard),'--num-shards','6']
        with (out/f'candidates_shard{shard}.log').open('w') as log:
            child=subprocess.Popen(command,env=worker_env,cwd=work,stdout=log,stderr=subprocess.STDOUT)
            status.update(status='candidates',pid=child.pid);write(path,status)
            if child.wait(timeout=1800):raise RuntimeError('Resumed candidate failed; inspect log')
        status.update(status='generation_complete',completed=time.time())
    except Exception as error:status.update(status='failed',error=str(error),completed=time.time())
    write(path,status);return status
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(worker,[(1,0),(2,6)]))
workers=[json.loads((out/f'worker_{i}.json').read_text()) for i in range(6)]
success=all(w['status']=='generation_complete' for w in workers)
write(out/'generation_status.json',dict(success=success,workers=workers,expected_images=144,optimizer_steps=0,resource_only_reschedule=True))
if not success:raise SystemExit(1)
