"""Two finite v81 workers only after complete local v80 verification."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path('/home/shy_23/iterative_visual_stage1_20261003/stage3_grounded_repair');OUT=ROOT/'reference_recomputed_train_v81'
PY='/home/shy_23/venvs/show_o2/bin/python'
ARMS=('recomputed_trainable4','recomputed_frozen4')
GPUS={0:'GPU-4a177c46-c73f-d963-8d61-b7fd90196038',4:'GPU-7c36f685-23c2-b1e5-dab9-b9a815b045da'}
def main():
    if OUT.exists():raise ValueError('Inspect existing v81, never duplicate')
    if not json.loads((ROOT/'v80_portable_verification.json').read_text())['passed']:raise ValueError('Require complete local v80 proof')
    for index,uuid in GPUS.items():
        q=subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
        if q[0].strip()!=uuid or int(q[1])>=2000:raise ValueError('GPU occupied or UUID changed, no other task stopped')
    OUT.mkdir();workers=[];logs=[]
    def save(phase,**kw):(OUT/'pipeline_status.json').write_text(json.dumps(dict(phase=phase,time=time.time(),workers=[dict(arm=a,pid=p.pid) for a,p in workers],**kw),indent=2))
    for arm,(index,uuid) in zip(ARMS,GPUS.items()):
        env=os.environ.copy();env.update(TORCHDYNAMO_DISABLE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONWARNINGS='ignore',CUDA_VISIBLE_DEVICES=uuid,PYTHONPATH=str(ROOT.parent)+':'+str(ROOT.parent/'deps'))
        log=(OUT/(arm+'.log')).open('w');logs.append(log)
        p=subprocess.Popen([PY,'-u','train_reference_recomputed.py','--arm',arm],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        workers.append((arm,p));save('training')
    deadline=time.monotonic()+10800
    while any(p.poll() is None for _,p in workers) and time.monotonic()<deadline:time.sleep(10)
    for _,p in workers:
        if p.poll() is None:p.terminate();p.wait(timeout=30)
    codes={a:p.returncode for a,p in workers}
    for log in logs:log.close()
    save('complete' if not any(codes.values()) else 'failed',exit_codes=codes,automatic_generation=False,automatic_val=False,controller_started=False)
if __name__=='__main__':main()
