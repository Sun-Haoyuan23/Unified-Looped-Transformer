"""Run capture -> candidates per UUID; GPU workers wait for existing jobs.

This orchestrates subprocesses, not autonomous agents. A stopped/failed worker
does not advance to the next phase. Project processes are never killed.
"""
import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path
from runtime import write_json

GPUS={0:'GPU-4a177c46-c73f-d963-8d61-b7fd90196038',
      1:'GPU-7ff2b0e0-b9c0-00f5-2ada-eb99a9d2a7a7',
      2:'GPU-d0ecf02c-9378-cb53-a381-8e4bd7fe6362',
      3:'GPU-388c574b-3a93-c8dd-774a-52cfda4f2e5d',
      4:'GPU-7c36f685-23c2-b1e5-dab9-b9a815b045da',
      6:'GPU-67e8fd78-bc55-3da3-30da-a6bc83ab7a23',
      7:'GPU-0dc80597-a9d6-bbc1-98b2-82aafd5a13bd'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--experiment',required=True)
    p.add_argument('--gpus',default='0,1,2,3,4,6,7')
    p.add_argument('--actions',default='baseline,generic_revisit,extra_ode,conditional_zero,semantic_probe,opposite_probe,prompt_repair,opposite_prompt')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--level',type=int,choices=(0,1,2),default=1)
    p.add_argument('--states',type=Path,help='Existing captures; skip capture phase')
    p.add_argument('--indices',default='30,35')
    args=p.parse_args()
    gpu_ids=[int(s) for s in args.gpus.split(',')]
    root=args.work/'stage3_grounded_repair'
    output=root/args.experiment;output.mkdir(parents=True,exist_ok=True)
    python='/home/shy_23/venvs/show_o2/bin/python'
    states=args.states or root/'states'/args.experiment
    def worker(shard,gpu):
        path=output/f'worker_{shard}.json'
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu],TORCHDYNAMO_DISABLE='1',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONWARNINGS='ignore',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(args.work/'deps'))
        status={'experiment_id':args.experiment,'shard':shard,'gpu_physical':gpu,
                'gpu_uuid':GPUS[gpu],'status':'waiting_for_gpu','started':time.time()}
        write_json(path,status)
        deadline=time.time()+1200
        while True:
            q=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used',
                '--format=csv,noheader,nounits'],capture_output=True,text=True)
            used=int(q.stdout.strip())
            limit=5000 if gpu==1 else 2000
            if used<limit:break
            if time.time()>deadline:raise TimeoutError('Existing GPU workload did not finish')
            time.sleep(20)
        commands=[] if args.states else [('capture',[python,str(root/'capture_repair_states.py'),'--work',str(args.work),
            '--tasks',str(args.tasks),'--output',str(states/f'shard{shard}'),
            '--shard',str(shard),'--num-shards',str(len(gpu_ids)),'--indices',args.indices])]
        candidate_cmd=[python,str(root/'build_repair_candidates.py'),'--work',str(args.work),
             '--states',str(states/f'shard{shard}'),'--output',str(output/f'shard{shard}'),
             '--actions',args.actions,'--level',str(args.level)]
        if args.checkpoint:candidate_cmd.extend(['--checkpoint',str(args.checkpoint)])
        commands.append(('candidates',candidate_cmd))
        try:
            for phase,cmd in commands:
                status.update(status=phase,pid=None);write_json(path,status)
                with (output/f'{phase}_shard{shard}.log').open('a') as log:
                    proc=subprocess.Popen(cmd,env=env,cwd=args.work,stdout=log,stderr=subprocess.STDOUT)
                    status['pid']=proc.pid;write_json(path,status)
                    code=proc.wait()
                if code:raise RuntimeError(f'{phase} exited {code}; see log')
            status.update(status='generation_complete',completed=time.time())
        except Exception as error:
            status.update(status='failed',error=str(error),completed=time.time())
        write_json(path,status)
        return status
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        results=list(pool.map(lambda x:worker(*x),enumerate(gpu_ids)))
    write_json(output/'generation_status.json',{'workers':results,
        'success':all(r['status']=='generation_complete' for r in results),
        'next':'official terminal evaluation; repair training only after H1 gate'})
    if any(r['status']=='failed' for r in results):raise SystemExit(1)


if __name__=='__main__':main()
