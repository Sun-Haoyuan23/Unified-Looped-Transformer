"""Run finite grounding diagnostics on healthy GPUs; preserve other jobs."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time

GPUS={0:'GPU-4a177c46-c73f-d963-8d61-b7fd90196038',1:'GPU-7ff2b0e0-b9c0-00f5-2ada-eb99a9d2a7a7',
      2:'GPU-d0ecf02c-9378-cb53-a381-8e4bd7fe6362',3:'GPU-388c574b-3a93-c8dd-774a-52cfda4f2e5d',
      4:'GPU-7c36f685-23c2-b1e5-dab9-b9a815b045da',6:'GPU-67e8fd78-bc55-3da3-30da-a6bc83ab7a23',
      7:'GPU-0dc80597-a9d6-bbc1-98b2-82aafd5a13bd'}


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--states',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    root=args.work/'stage3_grounded_repair'
    def worker(shard,gpu):
        status={'shard':shard,'gpu_physical':gpu,'gpu_uuid':GPUS[gpu],'status':'waiting_for_gpu','started':time.time()}
        path=args.output/f'worker_{shard}.json';write(path,status)
        try:
            deadline=time.time()+1200
            while True:
                q=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
                if int(q.stdout.strip())<(5000 if gpu==1 else 2000):break
                if time.time()>deadline:raise TimeoutError('GPU remained occupied; no process killed')
                time.sleep(20)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCHDYNAMO_DISABLE='1',PYTHONWARNINGS='ignore',PYTHONPATH=str(args.work/'deps'))
            cmd=['/home/shy_23/venvs/show_o2/bin/python',str(root/'diagnose_grounding.py'),'--work',str(args.work),
                '--states',*[str(s) for s in args.states],'--output',str(args.output),
                '--shard',str(shard),'--num-shards',str(len(GPUS))]
            with (args.output/f'worker_{shard}.log').open('w') as output:
                proc=subprocess.Popen(cmd,env=env,cwd=args.work,stdout=output,stderr=subprocess.STDOUT)
                status.update(status='running',pid=proc.pid);write(path,status)
                code=proc.wait(timeout=1800)
                if code:raise RuntimeError(f'Diagnostic exited {code}')
            status.update(status='complete',completed=time.time())
        except Exception as error:status.update(status='failed',error=str(error),completed=time.time())
        write(path,status);return status
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        results=list(pool.map(lambda pair:worker(*pair),enumerate(GPUS)))
    success=all(r['status']=='complete' for r in results)
    write(args.output/'diagnostic_status.json',{'success':success,'workers':results,'optimizer_steps':0})
    if not success:raise SystemExit(1)


if __name__=='__main__':main()
