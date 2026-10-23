"""Fixed four-arm state-action capacity experiment, then fixed TRAIN diagnosis."""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from candidate_provenance import file_hash, freeze_config
from training_pairs import validate_training_pairs
from launch_grounding_diagnostic import GPUS, write

w = Path('/home/shy_23/iterative_visual_stage1_20261003'); r = w/'stage3_grounded_repair'
out = r/'state_space_train_v44'; py = '/home/shy_23/venvs/show_o2/bin/python'
arms = ('head_only','outer','inner','joint')

def update(phase, **kw):
    write(out/'pipeline_status.json', dict(phase=phase,time=time.time(),controller_started=False,**kw))

def main():
    out.mkdir()
    try:
        proof = json.loads((r/'state_space_gradient_v44/pipeline_status.json').read_text())
        if (proof['phase']!='passed' or not proof['zero_intervention_latent_exact'] or
            not proof['shared_cores_exact'] or proof['base_parameter_gradients'] or proof['optimizer_updates']!=0 or
            proof['zero_arms']!=list(arms)):
            raise ValueError('Require all-arm actual zero and shared full-terminal-gradient proof')
        teacher = r/'repair_train_targets_v36'
        gate = validate_training_pairs(json.loads((teacher/'paired_report.json').read_text()),
                    list(map(json.loads,(teacher/'all_actions.jsonl').read_text().splitlines())))
        if gate['selected_teacher_states']!=3 or gate['preservation_states']!=35:
            raise ValueError('Strict TRAIN gate changed')
        names = ('run_state_space_training.py','train_state_space_operator.py','state_space_revisit.py',
                 'test_state_space_revisit.py','check_state_space_gradient.py','runtime.py','evidence_encoder.py','training_pairs.py')
        freeze_config(out/'config.json',dict(training_gate=gate,steps_per_arm=100,width=64,level=2,window=1,lr=1e-4,
            source_sha256={n:file_hash(r/n) for n in names},teacher_report_sha256=file_hash(teacher/'paired_report.json'),
            teacher_records_sha256=file_hash(teacher/'all_actions.jsonl'),gradient_proof_sha256=file_hash(r/'state_space_gradient_v44/pipeline_status.json'),
            state_sampling_seed=20261004,common_initialization_seed=20261005,preservation_every=3,
            checkpoint_selection='fixed final step100, frozen before inference',controller=False,val_images=0,
            rationale='Successful teacher changes physical midflow state; learn that action through shared recurrent token processing instead of only perturbing integrated velocity',
            comparison='head-only matched readout control plus outer/inner/joint; unequal costs; no equal-FLOPs or total-parameter reduction claim'))
        (out/'source').mkdir()
        for n in names: shutil.copy2(r/n,out/'source'/n)
        with (out/'unit_tests.log').open('w') as log:
            subprocess.run([py,'-m','unittest','test_state_space_revisit.py'],cwd=r,stdout=log,stderr=subprocess.STDOUT,check=True)
        update('training',steps_per_arm=100)
        def worker(arm,gpu):
            state=dict(arm=arm,gpu=gpu,status='waiting_for_gpu'); path=out/(arm+'.json'); write(path,state)
            try:
                deadline=time.time()+1800
                while int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())>=2000:
                    if time.time()>deadline: raise TimeoutError('GPU occupied; no other task interrupted')
                    time.sleep(20)
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                    TORCHDYNAMO_DISABLE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(w/'deps'),PYTHONWARNINGS='ignore')
                command=[py,str(r/'train_state_space_operator.py'),'--work',str(w),'--states',str(r/'states/repair_train_targets_v36'),
                         '--records',str(teacher/'all_actions.jsonl'),'--paired-report',str(teacher/'paired_report.json'),
                         '--output',str(out/arm),'--steps','100','--width','64','--window','1','--arm',arm]
                with (out/(arm+'.log')).open('w') as log:
                    p=subprocess.Popen(command,cwd=w,env=env,stdout=log,stderr=subprocess.STDOUT)
                    state.update(pid=p.pid,status='training');write(path,state)
                    if p.wait(timeout=3600):raise RuntimeError(arm+' training failed')
                state.update(status='complete')
            except Exception as error:state.update(status='failed',error=str(error))
            write(path,state);return state
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(lambda q:worker(*q),zip(arms,(0,4,6,7))))
        if any(q['status']!='complete' for q in results):raise ValueError('One or more arms failed')
        configs={a:json.loads((out/a/'training_config.json').read_text()) for a in arms}
        initial={a:c['adapter_initialization_hashes'] for a,c in configs.items()}
        if initial['head_only']!=initial['outer']:raise ValueError('Head-only/outer initialization mismatch')
        for arm in ('outer','inner'):
            if any(initial['joint'][k]!=v for k,v in initial[arm].items()):raise ValueError('Joint branch initialization mismatch')
        sequences={a:[(q['state_id'],q['preservation_step']) for q in map(json.loads,(out/a/'train.jsonl').read_text().splitlines())] for a in arms}
        if any(len(q)!=100 for q in sequences.values()) or any(q!=sequences['outer'] for q in sequences.values()):
            raise ValueError('Training update/sample sequence mismatch')
        write(out/'matched_initialization.json',dict(passed=True,all_common_and_branch_tensors_matched=True,state_sampling_exact=True))
        update('training_complete',optimizer_updates=400,val_images=0)
        # Finite fixed TRAIN diagnosis only. Failed training never starts inference.
        with (out/'trainfit_pipeline.log').open('w') as log:
            subprocess.run([py,str(r/'run_state_space_trainfit.py')],cwd=w,stdout=log,stderr=subprocess.STDOUT,check=True)
        update('complete',optimizer_updates=400,trainfit_images=120,val_images=0,
               next='Assess same-state semantics/quality/preservation and head-only evidence controls before independent val')
    except Exception as error:update('failed',error=str(error));raise

if __name__=='__main__':main()
