"""Finite four GPU v78 launch only after complete local v77 verification."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path('/home/shy_23/iterative_visual_stage1_20261003/stage3_grounded_repair');OUT=ROOT/'process_four_loop_train_v78'
PY='/home/shy_23/venvs/show_o2/bin/python'
ARMS=('anchor_trainable4','anchor_frozen4','plain_trainable4','no_loop')
GPUS={0:'GPU-4a177c46-c73f-d963-8d61-b7fd90196038',4:'GPU-7c36f685-23c2-b1e5-dab9-b9a815b045da',6:'GPU-67e8fd78-bc55-3da3-30da-a6bc83ab7a23',7:'GPU-0dc80597-a9d6-bbc1-98b2-82aafd5a13bd'}
def main():
    if OUT.exists():raise ValueError('Inspect existing v78, never duplicate')
    proof=json.loads((ROOT/'v77_portable_verification.json').read_text())
    if not proof['passed']:raise ValueError('Require complete local v77 verification')
    for index,uuid in GPUS.items():
        q=subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
        if q[0].strip()!=uuid or int(q[1])>=2000:raise ValueError('GPU occupied or UUID changed, no other task stopped')
    OUT.mkdir();workers=[]
    def save(phase,**kw):(OUT/'pipeline_status.json').write_text(json.dumps(dict(phase=phase,time=time.time(),workers=[dict(arm=a,pid=p.pid) for a,p in workers],**kw),indent=2))
    for arm,(index,uuid) in zip(ARMS,GPUS.items()):
        env=os.environ.copy();env.update(TORCHDYNAMO_DISABLE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONWARNINGS='ignore',CUDA_VISIBLE_DEVICES=uuid,PYTHONPATH=str(ROOT.parent)+':'+str(ROOT.parent/'deps'))
        log=(OUT/(arm+'.log')).open('w');p=subprocess.Popen([PY,'-u','train_process_four_loop.py','--arm',arm],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        workers.append((arm,p));save('training')
    codes={a:p.wait() for a,p in workers};save('complete' if not any(codes.values()) else 'failed',exit_codes=codes,automatic_generation=False,automatic_val=False,controller_started=False)
if __name__=='__main__':main()
