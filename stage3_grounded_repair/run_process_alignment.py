"""One bounded v79 read-only worker, no automatic training or scoring."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path('/home/shy_23/iterative_visual_stage1_20261003/stage3_grounded_repair')
PY='/home/shy_23/venvs/show_o2/bin/python';UUID='GPU-0dc80597-a9d6-bbc1-98b2-82aafd5a13bd';STATUS=ROOT/'transfers/process_alignment_v79_launcher_status.json'


def main():
    if STATUS.exists() or (ROOT/'process_four_loop_alignment_v79').exists():raise ValueError('Inspect existing v79, never duplicate')
    env=os.environ.copy();env.update(TORCHDYNAMO_DISABLE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONWARNINGS='ignore',PYTHONPATH=str(ROOT.parent)+':'+str(ROOT.parent/'deps'))
    with (ROOT/'transfers/process_alignment_v79_unit_tests.log').open('w') as log:subprocess.run([PY,'-m','unittest','test_process_alignment_diagnostics'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    q=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
    if q[0].strip()!=UUID or int(q[1])>=2000:raise ValueError('GPU7 occupied or UUID changed')
    env['CUDA_VISIBLE_DEVICES']=UUID
    def save(phase,**kw):STATUS.write_text(json.dumps(dict(phase=phase,time=time.time(),optimizer_updates=0,new_scoring_images=0,**kw),indent=2))
    with (ROOT/'transfers/process_alignment_v79_worker.log').open('w') as log:
        p=subprocess.Popen([PY,'-u','check_process_alignment.py'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);save('running',worker_pid=p.pid)
        try:code=p.wait(timeout=3600)
        except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=30);save('failed',error='own worker timeout');return
    save('complete' if code==0 else 'failed',worker_pid=p.pid,exit_code=code)


if __name__=='__main__':main()
