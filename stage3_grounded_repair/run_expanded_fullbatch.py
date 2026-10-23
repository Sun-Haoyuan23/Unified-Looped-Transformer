"""Finite matched physical-only optimization; no quality expansion."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path('/home/shy_23/iterative_visual_stage1_20261003/stage3_grounded_repair');PY='/home/shy_23/venvs/show_o2/bin/python'
GPUS={0:'GPU-4a177c46-c73f-d963-8d61-b7fd90196038',4:'GPU-7c36f685-23c2-b1e5-dab9-b9a815b045da',6:'GPU-67e8fd78-bc55-3da3-30da-a6bc83ab7a23'}
OUT=ROOT/'expanded_fullbatch_capacity_v74'


def main():
    OUT.mkdir(exist_ok=False)
    def status(phase,**kw):(OUT/'pipeline_status.json').write_text(json.dumps(dict(phase=phase,time=time.time(),**kw),indent=2)+'\n')
    env=os.environ.copy();env.update(TORCHDYNAMO_DISABLE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        PYTHONWARNINGS='ignore',PYTHONPATH=str(ROOT.parent)+':'+str(ROOT.parent/'deps'))
    status('testing')
    with (OUT/'unit_tests.log').open('w') as log:
        test=subprocess.run([PY,'-m','unittest','test_expanded_fullbatch'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=60)
    if test.returncode:status('failed',stage='new_cpu_tests',exit_code=test.returncode);return
    for gpu,uuid in GPUS.items():
        q=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
        if q[0].strip()!=uuid or int(q[1])>=2000:status('failed',stage='resource_check',gpu=gpu);return
    workers=[];logs=[]
    for (gpu,uuid),arm in zip(GPUS.items(),('no_loop','frozen_loop','trainable_loop')):
        child=env.copy();child['CUDA_VISIBLE_DEVICES']=uuid;log=(OUT/(arm+'.log')).open('w');logs.append(log)
        p=subprocess.Popen([PY,'-u','train_expanded_fullbatch.py','--arm',arm],cwd=ROOT,env=child,stdout=log,stderr=subprocess.STDOUT)
        workers.append((arm,p))
    status('optimizing',workers=[dict(arm=a,pid=p.pid) for a,p in workers])
    deadline=time.monotonic()+7200
    while any(p.poll() is None for a,p in workers) and time.monotonic()<deadline:time.sleep(5)
    for a,p in workers:
        if p.poll() is None:p.terminate();p.wait(timeout=30)
    for log in logs:log.close()
    status('complete' if all(p.returncode==0 for a,p in workers) else 'failed',workers=[dict(arm=a,pid=p.pid,exit_code=p.returncode) for a,p in workers],
        no_automatic_endpoint_or_val_or_controller=True)


if __name__=='__main__':main()
