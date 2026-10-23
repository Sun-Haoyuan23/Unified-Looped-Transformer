"""One finite head-capacity pipeline; never launches generation or controller."""
import json,os,subprocess,time
from pathlib import Path

ROOT=Path('/home/shy_23/iterative_visual_stage1_20261003/stage3_grounded_repair')
PY='/home/shy_23/venvs/show_o2/bin/python'
STATUS=ROOT/'transfers/nonlinear_readout_capacity_v60_launcher_status.json'


def save(phase,**kw):
    STATUS.write_text(json.dumps(dict(phase=phase,time=time.time(),**kw),indent=2)+'\n')


def main():
    if STATUS.exists() or (ROOT/'nonlinear_readout_capacity_v60').exists():
        raise ValueError('Existing v60 run; inspect instead of relaunching')
    env=os.environ.copy();env.update(TORCHDYNAMO_DISABLE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONWARNINGS='ignore')
    save('testing')
    with (ROOT/'transfers/nonlinear_readout_capacity_v60_unit_tests.log').open('w') as log:
        result=subprocess.run([PY,'-m','unittest','test_nonlinear_token_readout'],cwd=ROOT,env=env,
            stdout=log,stderr=subprocess.STDOUT,timeout=60)
    if result.returncode:save('failed',stage='unit_tests',exit_code=result.returncode);return
    env['CUDA_VISIBLE_DEVICES']='GPU-4a177c46-c73f-d963-8d61-b7fd90196038'
    env['PYTHONPATH']=str(ROOT.parent)+':'+str(ROOT.parent/'deps')
    with (ROOT/'transfers/nonlinear_readout_capacity_v60_worker.log').open('w') as log:
        worker=subprocess.Popen([PY,'-u','train_nonlinear_readout_capacity.py'],cwd=ROOT,env=env,
            stdout=log,stderr=subprocess.STDOUT)
        save('running',worker_pid=worker.pid)
        try:code=worker.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            worker.terminate();worker.wait(timeout=30);save('failed',stage='worker_timeout',worker_pid=worker.pid);return
    save('complete' if code==0 else 'failed',worker_pid=worker.pid,exit_code=code)


if __name__=='__main__':main()
