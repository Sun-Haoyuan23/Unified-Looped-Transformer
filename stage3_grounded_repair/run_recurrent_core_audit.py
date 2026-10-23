"""Finite read-only v48 audit, one worker per currently idle healthy GPU."""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from candidate_provenance import file_hash, freeze_config
from launch_grounding_diagnostic import GPUS, write

WORK = Path('/home/shy_23/iterative_visual_stage1_20261003')
ROOT = WORK / 'stage3_grounded_repair'
OUT = ROOT / 'recurrent_core_audit_v48'
PYTHON = '/home/shy_23/venvs/show_o2/bin/python'
STUDIES = ('outer_ilsd', 'inner_ilsd', 'joint_ilsd', 'joint_terminal')


def main():
    OUT.mkdir()
    def status(phase, **kwargs):
        write(OUT/'pipeline_status.json', dict(phase=phase, time=time.time(), optimizer_updates=0,
            new_scored_images=0, controller_started=False, **kwargs))
    try:
        for name in ('recurrent_train_v46', 'recurrent_trainfit_v47'):
            if json.loads((ROOT/name/'pipeline_status.json').read_text())['phase'] != 'complete':
                raise ValueError('Previous fixed training/scoring not complete')
        names = ('audit_recurrent_core_contribution.py','run_recurrent_core_audit.py',
            'trainable_recurrent_state.py','recurrent_training_protocol.py','state_space_revisit.py',
            'runtime.py','training_pairs.py','evidence_encoder.py','candidate_provenance.py')
        freeze_config(OUT/'config.json', dict(source_sha256={n:file_hash(ROOT/n) for n in names},
            cohort_sha256=file_hash(ROOT/'recurrent_trainfit_v47/cohort.json'),
            checkpoint_sha256={a:file_hash(ROOT/'recurrent_train_v46'/a/'operator_step100.pt') for a in STUDIES},
            actions=['learned_core_replay','trained_readout_frozen_core'], budget=2,
            states_per_study=6, studies=STUDIES, optimizer_updates=0, scored_images=0,
            note='Read-only inference dependence audit; LoRA-off is not matched retraining'))
        (OUT/'source').mkdir()
        for n in names:
            shutil.copy2(ROOT/n,OUT/'source'/n)
        status('auditing', expected_strict_replays=24, expected_counterfactual_rollouts=24)
        def worker(item):
            study, gpu = item
            worker_path = OUT/(study+'.json')
            row = dict(study=study,gpu_physical=gpu,gpu_uuid=GPUS[gpu],status='checking_gpu')
            write(worker_path,row)
            try:
                q = subprocess.check_output(['nvidia-smi','-i',str(gpu),
                    '--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
                if q[0].strip() != GPUS[gpu] or int(q[1].strip()) >= 2000:
                    raise RuntimeError('GPU occupied or UUID mismatch; no other task interrupted')
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                    TORCHDYNAMO_DISABLE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                    PYTHONWARNINGS='ignore',PYTHONPATH=str(WORK/'deps'))
                command = [PYTHON,str(ROOT/'audit_recurrent_core_contribution.py'),'--work',str(WORK),
                    '--states',str(ROOT/'states/repair_train_targets_v36'),
                    '--teacher-root',str(ROOT/'repair_train_targets_v36'),
                    '--cohort',str(ROOT/'recurrent_trainfit_v47/cohort.json'),
                    '--checkpoint',str(ROOT/'recurrent_train_v46'/study/'operator_step100.pt'),
                    '--reference',str(ROOT/'recurrent_trainfit_v47'/study),
                    '--output',str(OUT/study),'--study',study]
                with (OUT/(study+'.log')).open('w') as log:
                    proc = subprocess.Popen(command,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
                    row.update(pid=proc.pid,status='auditing')
                    write(worker_path,row)
                    if proc.wait(timeout=3600):
                        raise RuntimeError('Audit worker failed; read log')
                row.update(status='complete')
            except Exception as error:
                row.update(status='failed',error=str(error))
            write(worker_path,row)
            return row
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(worker,zip(STUDIES,(0,4,6,7))))
        if any(r['status'] != 'complete' for r in rows):
            raise RuntimeError('One or more read-only workers failed')
        status('complete',strict_replays=24,counterfactual_rollouts=24)
    except Exception as error:
        status('failed',error=str(error))
        raise


if __name__ == '__main__':
    main()
