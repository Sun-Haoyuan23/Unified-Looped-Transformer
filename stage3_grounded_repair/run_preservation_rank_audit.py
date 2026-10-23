"""Single finite, identified v59 worker with a hard timeout; no automatic retries."""
import os
from pathlib import Path
import subprocess
import time
from launch_grounding_diagnostic import GPUS,write

WORK=Path('/home/shy_23/iterative_visual_stage1_20261003')
ROOT=WORK/'stage3_grounded_repair'
PYTHON='/home/shy_23/venvs/show_o2/bin/python'
STATUS=ROOT/'transfers/preservation_rank_audit_v59_launcher_status.json'


def main():
    if (ROOT/'preservation_rank_audit_v59').exists() or STATUS.exists():
        raise ValueError('Frozen v59 already exists; do not duplicate')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPUS[0],TORCHDYNAMO_DISABLE='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        PYTHONWARNINGS='ignore',PYTHONPATH=str(WORK/'deps'))
    row=dict(phase='unit_tests',start_time=time.time(),hard_timeout_s=3600,optimizer_updates=0,new_scoring_images=0)
    write(STATUS,row)
    try:
        with (ROOT/'transfers/preservation_geometry_v59_unit_tests.log').open('w') as log:
            subprocess.run([PYTHON,'-m','unittest','test_preservation_geometry','-v'],cwd=ROOT,env=env,
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
        with (ROOT/'transfers/preservation_rank_audit_v59_worker.log').open('w') as log:
            process=subprocess.Popen([PYTHON,str(ROOT/'audit_preservation_rank.py')],cwd=WORK,env=env,
                stdout=log,stderr=subprocess.STDOUT)
            row.update(phase='running',worker_pid=process.pid);write(STATUS,row)
            try:
                code=process.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                # This handle identifies only this launcher's own worker.
                row.update(phase='hard_timeout',time=time.time());write(STATUS,row)
                process.terminate()
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                raise
        if code:raise RuntimeError('v59 worker failed; retained evidence, no automatic retry')
        row.update(phase='complete',end_time=time.time());write(STATUS,row)
    except Exception as error:
        row.update(phase='failed',error=str(error),end_time=time.time());write(STATUS,row);raise


if __name__=='__main__':main()
