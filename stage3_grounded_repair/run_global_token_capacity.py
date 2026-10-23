"""Two resource-checked capacity workers, not a generation or core-training run."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path('/home/shy_23/iterative_visual_stage1_20261003/stage3_grounded_repair');PY='/home/shy_23/venvs/show_o2/bin/python'
OUT=ROOT/'global_token_capacity_v66'
GPUS={0:'GPU-4a177c46-c73f-d963-8d61-b7fd90196038',4:'GPU-7c36f685-23c2-b1e5-dab9-b9a815b045da'}
def main():
    if OUT.exists():raise ValueError('Inspect existing v66; do not duplicate')
    for i,u in GPUS.items():
        q=subprocess.check_output(['nvidia-smi','-i',str(i),'--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
        if q[0].strip()!=u or int(q[1])>=2000:raise ValueError('GPU occupied/UUID changed')
    OUT.mkdir();env=os.environ.copy();env.update(TORCHDYNAMO_DISABLE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
       OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONWARNINGS='ignore',PYTHONPATH=str(ROOT.parent)+':'+str(ROOT.parent/'deps'))
    def status(phase,**kw):(OUT/'pipeline_status.json').write_text(json.dumps(dict(phase=phase,time=time.time(),new_scoring_images=0,
       shared_core_optimizer_updates=0,automatic_generation=False,automatic_val=False,controller_started=False,**kw),indent=2)+'\n')
    status('unit_tests')
    with (OUT/'unit_tests.log').open('w') as log:
        t=subprocess.run([PY,'-m','unittest','test_global_token_readout'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
    if t.returncode:status('failed',stage='unit_tests');return
    workers=[]
    for mode,gpu in [('loop2',0),('no_loop',4)]:
        e=env.copy();e['CUDA_VISIBLE_DEVICES']=GPUS[gpu]
        with (OUT/(mode+'.log')).open('w') as log:p=subprocess.Popen([PY,'-u','train_global_token_capacity.py','--mode',mode],cwd=ROOT,env=e,stdout=log,stderr=subprocess.STDOUT)
        workers.append((mode,p))
    status('training_head',workers=[dict(mode=m,pid=p.pid) for m,p in workers])
    codes={m:p.wait() for m,p in workers}
    if set(codes.values())!={0}:status('failed',exit_codes=codes);return
    init=[json.loads((OUT/m/'initialization.json').read_text())['trainable_tensor_sha256'] for m,p in workers]
    if init[0]!=init[1]:status('failed',error='trainable head initialization mismatch');return
    results={m:json.loads((OUT/m/'capacity_report.json').read_text()) for m,p in workers}
    (OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    status('complete',exit_codes=codes,total_head_optimizer_updates=2000,matched_trainable_head_initialization=True)
if __name__=='__main__':main()
