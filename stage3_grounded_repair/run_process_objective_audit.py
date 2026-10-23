"""Finite four-arm read-only diagnosis. No training or scoring is started."""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from candidate_provenance import file_hash, freeze_config
from launch_grounding_diagnostic import GPUS, write

WORK=Path('/home/shy_23/iterative_visual_stage1_20261003');ROOT=WORK/'stage3_grounded_repair'
OUT=ROOT/'process_objective_audit_v54';PYTHON='/home/shy_23/venvs/show_o2/bin/python'
ARMS=[('joint_control',0),('outer_process',4),('inner_process',6),('joint_process',7)]


def main():
    OUT.mkdir()
    def status(phase,**kw):write(OUT/'pipeline_status.json',dict(phase=phase,time=time.time(),optimizer_updates=0,new_scoring_images=0,controller=False,**kw))
    try:
        if json.loads((ROOT/'process_recurrent_trainfit_v53/pipeline_status.json').read_text())['phase']!='complete':
            raise ValueError('Require completed v53 before diagnosis')
        names=['audit_process_objectives.py','run_process_objective_audit.py','process_supervision.py',
            'process_training_targets.py','recurrent_training_protocol.py','trainable_recurrent_state.py',
            'state_space_revisit.py','runtime.py','evidence_encoder.py','training_pairs.py','candidate_provenance.py']
        freeze_config(OUT/'config.json',dict(source_sha256={n:file_hash(ROOT/n) for n in names},
            checkpoint_sha256={a:file_hash(ROOT/'process_recurrent_train_v52'/a/'operator_step100.pt') for a,g in ARMS},
            cohort_sha256=file_hash(ROOT/'process_recurrent_trainfit_v53/cohort.json'),
            arms=ARMS,losses=['terminal at step100','physical state','normalized hidden'],
            strict_v53_endpoint_replays=48,optimizer_updates=0,new_scoring_images=0,FLOPs=None))
        (OUT/'source').mkdir()
        for n in names:shutil.copy2(ROOT/n,OUT/'source'/n)
        status('running')
        def worker(item):
            arm,gpu=item;row=dict(arm=arm,gpu=gpu,status='checking_gpu')
            try:
                q=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
                if q[0].strip()!=GPUS[gpu] or int(q[1].strip())>=2000:raise RuntimeError('GPU occupied/identity mismatch; others untouched')
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu],TORCHDYNAMO_DISABLE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONWARNINGS='ignore',PYTHONPATH=str(WORK/'deps'))
                with (OUT/(arm+'.log')).open('w') as log:
                    p=subprocess.Popen([PYTHON,str(ROOT/'audit_process_objectives.py'),'--arm',arm],cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
                    row.update(pid=p.pid,status='running');write(OUT/(arm+'.json'),row)
                    if p.wait(timeout=1800):raise RuntimeError('Read-only diagnostic failed')
                row.update(status='complete')
            except Exception as error:row.update(status='failed',error=str(error))
            write(OUT/(arm+'.json'),row);return row
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(worker,ARMS))
        if any(q['status']!='complete' for q in rows):raise ValueError('One or more diagnostic arms failed')
        status('complete',strict_endpoint_replays=48,states_per_arm=6,next='Analyze state/hidden/terminal direction conflict; no automatic training/val/controller')
    except Exception as error:status('failed',error=str(error));raise


if __name__=='__main__':main()
