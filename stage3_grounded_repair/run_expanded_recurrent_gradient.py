"""Bounded v71 CPU tests and real model proof on verified idle GPU7."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path('/home/shy_23/iterative_visual_stage1_20261003/stage3_grounded_repair')
PY='/home/shy_23/venvs/show_o2/bin/python';STATUS=ROOT/'transfers/expanded_recurrent_gradient_v71_launcher_status.json'
UUID='GPU-0dc80597-a9d6-bbc1-98b2-82aafd5a13bd'

def main():
    if STATUS.exists() or (ROOT/'expanded_recurrent_gradient_v71').exists():raise ValueError('Inspect v71; do not relaunch')
    q=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
    if q[0].strip()!=UUID or int(q[1])>=2000:raise ValueError('GPU7 occupied or UUID mismatch')
    env=os.environ.copy();env.update(TORCHDYNAMO_DISABLE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        PYTHONWARNINGS='ignore',PYTHONPATH=str(ROOT.parent)+':'+str(ROOT.parent/'deps'))
    def save(phase,**kw):STATUS.write_text(json.dumps(dict(phase=phase,time=time.time(),**kw),indent=2)+'\n')
    with (ROOT/'transfers/expanded_recurrent_gradient_v71_tests.log').open('w') as log:
        result=subprocess.run([PY,'-m','unittest','test_expanded_process_corpus'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
    if result.returncode:save('failed',stage='CPU_tests');return
    env['CUDA_VISIBLE_DEVICES']=UUID
    with (ROOT/'transfers/expanded_recurrent_gradient_v71_worker.log').open('w') as log:
        worker=subprocess.Popen([PY,'-u','check_expanded_recurrent_gradient.py'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        save('running',worker_pid=worker.pid)
        try:code=worker.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            worker.terminate();worker.wait(timeout=30);save('failed',stage='own_worker_timeout');return
    save('complete' if code==0 else 'failed',worker_pid=worker.pid,exit_code=code)

if __name__=='__main__':main()
