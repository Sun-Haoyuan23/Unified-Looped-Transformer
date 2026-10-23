"""Bounded v72 three-arm training; only starts after real v71 passes."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path('/home/shy_23/iterative_visual_stage1_20261003/stage3_grounded_repair')
PY='/home/shy_23/venvs/show_o2/bin/python';OUT=ROOT/'expanded_recurrent_train_v72'
ARMS=('no_loop','frozen_loop','trainable_loop')
GPUS={0:'GPU-4a177c46-c73f-d963-8d61-b7fd90196038',4:'GPU-7c36f685-23c2-b1e5-dab9-b9a815b045da',6:'GPU-67e8fd78-bc55-3da3-30da-a6bc83ab7a23'}

def main():
    if OUT.exists():raise ValueError('Inspect existing v72; do not relaunch')
    proof=json.loads((ROOT/'expanded_recurrent_gradient_v71/pipeline_status.json').read_text())
    if proof['phase']!='passed' or not proof['all_three_initializations_equal']:raise ValueError('Require actual matched v71 proof')
    for index,uuid in GPUS.items():
        q=subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
        if q[0].strip()!=uuid or int(q[1])>=2000:raise ValueError('Occupied GPU or changed UUID; no other tasks stopped')
    OUT.mkdir();workers=[]
    def save(phase,**kw):
        (OUT/'pipeline_status.json').write_text(json.dumps(dict(phase=phase,time=time.time(),workers=[dict(arm=a,pid=p.pid) for a,p in workers],**kw),indent=2)+'\n')
    for arm,(index,uuid) in zip(ARMS,GPUS.items()):
        env=os.environ.copy();env.update(TORCHDYNAMO_DISABLE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
            PYTHONWARNINGS='ignore',CUDA_VISIBLE_DEVICES=uuid,PYTHONPATH=str(ROOT.parent)+':'+str(ROOT.parent/'deps'))
        log=(OUT/(arm+'.log')).open('w')
        p=subprocess.Popen([PY,'-u','train_expanded_recurrent_operator.py','--arm',arm],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        workers.append((arm,p));save('training')
    codes={a:p.wait() for a,p in workers}
    save('complete' if not any(codes.values()) else 'failed',exit_codes=codes,automatic_generation=False,automatic_val=False,controller_started=False)

if __name__=='__main__':main()
