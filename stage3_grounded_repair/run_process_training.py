"""Fixed matched process/control training, then finite TRAIN diagnosis only."""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from candidate_provenance import file_hash,freeze_config
from launch_grounding_diagnostic import GPUS,write

WORK=Path('/home/shy_23/iterative_visual_stage1_20261003')
ROOT=WORK/'stage3_grounded_repair'
OUT=ROOT/'process_recurrent_train_v52'
PYTHON='/home/shy_23/venvs/show_o2/bin/python'
ARMS=(('joint_control','joint_ilsd','terminal_control',0),
      ('outer_process','outer_ilsd','process',4),
      ('inner_process','inner_ilsd','process',6),
      ('joint_process','joint_ilsd','process',7))


def main():
    OUT.mkdir()
    def status(phase,**kw):
        write(OUT/'pipeline_status.json',dict(phase=phase,time=time.time(),controller_started=False,**kw))
    try:
        proof=ROOT/'process_shared_gradient_v51'
        result=json.loads((proof/'pipeline_status.json').read_text())
        if (result['phase']!='passed' or result['optimizer_updates']!=0 or
                result['zero_terminal_budgets_exact']!=[1,2] or result['base_parameter_gradients'] or
                not result['shared_core_hash_unchanged']):
            raise ValueError('Actual process interface proof required')
        for name,digest in json.loads((proof/'config.json').read_text())['configuration']['source_sha256'].items():
            if file_hash(ROOT/name)!=digest:raise ValueError('Proven source changed: '+name)
        portable=json.loads((ROOT/'v49_v50_portable_verification.json').read_text())
        if not portable['passed'] or not portable['v50_portable_paired_report_exact']:
            raise ValueError('Require original teacher process/quality portable verification')
        names=('run_process_training.py','run_process_trainfit.py','train_process_recurrent_operator.py',
            'process_supervision.py','process_training_targets.py','test_process_supervision.py',
            'check_process_gradient.py','trainable_recurrent_state.py','recurrent_training_protocol.py',
            'state_space_revisit.py','runtime.py','evidence_encoder.py','training_pairs.py','candidate_provenance.py')
        freeze_config(OUT/'config.json',dict(source_sha256={n:file_hash(ROOT/n) for n in names},
            proof_sha256=file_hash(proof/'pipeline_status.json'),teacher_portable_sha256=file_hash(ROOT/'v49_v50_portable_verification.json'),
            aligned_target_report_sha256=file_hash(ROOT/'teacher_additive_reachability_v50/paired_report.json'),
            arms=ARMS,steps_per_arm=100,lr=1e-4,rank=4,width=64,window=1,loops=[1,2],
            init_seed=20261005,state_sampling_seed=20261004,preserve_every=3,
            same_deployment_forward=True,added_process_parameters=0,
            control='joint_control/process identical forward/initialization/targets/sampling/nested terminal loss; process adds state and hidden losses',
            teacher_targets='v50 residual-interface endpoints and same-state original+gold_delta hidden targets',
            new_control_reason='v50 endpoints differ numerically from v46 targets; fresh matched control is required, not a rerun of v46',
            checkpoint_selection='fixed final step100, no val/controller',FLOPs=None))
        (OUT/'source').mkdir()
        for n in names:shutil.copy2(ROOT/n,OUT/'source'/n)
        shutil.copy2(ROOT/'transfers/process_v51_unit_tests.log',OUT/'unit_tests.log')
        status('training',planned_optimizer_updates=400)
        def worker(item):
            name,study,objective,gpu=item
            row=dict(arm=name,study=study,objective=objective,gpu=gpu,status='checking_gpu')
            path=OUT/(name+'.json');write(path,row)
            try:
                q=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.used',
                    '--format=csv,noheader,nounits'],text=True).strip().split(',')
                if q[0].strip()!=GPUS[gpu] or int(q[1].strip())>=2000:
                    raise RuntimeError('GPU occupied/identity mismatch; no other task interrupted')
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu],TORCHDYNAMO_DISABLE='1',
                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                    PYTHONWARNINGS='ignore',PYTHONPATH=str(WORK/'deps'))
                cmd=[PYTHON,str(ROOT/'train_process_recurrent_operator.py'),'--work',str(WORK),
                    '--output',str(OUT/name),'--study',study,'--objective',objective]
                with (OUT/(name+'.log')).open('w') as log:
                    proc=subprocess.Popen(cmd,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
                    row.update(pid=proc.pid,status='training');write(path,row)
                    if proc.wait(timeout=5400):raise RuntimeError('Fixed training arm failed')
                row.update(status='complete')
            except Exception as error:
                row.update(status='failed',error=str(error))
            write(path,row);return row
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            rows=list(pool.map(worker,ARMS))
        if any(q['status']!='complete' for q in rows):raise ValueError('At least one arm failed; no automatic trainfit')
        configs={a:json.loads((OUT/a/'training_config.json').read_text()) for a,_,_,_ in ARMS}
        initial={a:q['adapter_initialization_hashes'] for a,q in configs.items()}
        if initial['joint_control']!=initial['joint_process']:
            raise ValueError('Matched joint initialization differs')
        for arm in ('outer_process','inner_process'):
            if any(initial['joint_process'][k]!=v for k,v in initial[arm].items()):
                raise ValueError('Common/branch tensor initialization differs')
        sequences={a:[(q['state_id'],q['preservation_step']) for q in map(json.loads,(OUT/a/'train.jsonl').read_text().splitlines())]
            for a,_,_,_ in ARMS}
        if any(len(v)!=100 or v!=sequences['joint_control'] for v in sequences.values()):
            raise ValueError('Fixed100 sampling not matched')
        write(OUT/'matched_initialization.json',dict(passed=True,all_common_and_branch_tensors_matched=True,state_sampling_exact=True))
        status('training_complete',optimizer_updates=400)
        with (OUT/'trainfit_pipeline.log').open('w') as log:
            subprocess.run([PYTHON,str(ROOT/'run_process_trainfit.py')],cwd=WORK,stdout=log,stderr=subprocess.STDOUT,check=True)
        status('complete',optimizer_updates=400,trainfit_scoring_rows=144,validation_images=0)
    except Exception as error:
        status('failed',error=str(error));raise


if __name__=='__main__':main()
