"""Outer TRAIN fit pilot after actual zero/gradient proof; no val/controller."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from runtime import write_json
from candidate_provenance import file_hash,freeze_config
from training_pairs import validate_training_pairs
from launch_grounding_diagnostic import GPUS

w=Path('/home/shy_23/iterative_visual_stage1_20261003');r=w/'stage3_grounded_repair'
out=r/'outer_capacity_v33';out.mkdir()
py='/home/shy_23/venvs/show_o2/bin/python'
teacher=r/'oracle_encoded_correlation_train24x2_v23';states=r/'states/train_pilot_minimal_v14'

def update(phase,**kw):write_json(out/'pipeline_status.json',dict(phase=phase,time=time.time(),controller_started=False,**kw))

def run(cmd,label):
    deadline=time.time()+1800
    while int(subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())>=2000:
        if time.time()>deadline:raise TimeoutError('GPU occupied; no other process interrupted')
        time.sleep(20)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPUS[4],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TORCHDYNAMO_DISABLE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(w/'deps'),PYTHONWARNINGS='ignore')
    with (out/(label+'.log')).open('w') as stream:
        p=subprocess.Popen(cmd,cwd=w,env=env,stdout=stream,stderr=subprocess.STDOUT)
        write_json(out/(label+'.json'),dict(pid=p.pid,gpu=4,status='running'))
        if p.wait(timeout=1800):raise RuntimeError(label+' failed')
    write_json(out/(label+'.json'),dict(pid=p.pid,gpu=4,status='complete'))

try:
    proof=json.loads((r/'outer_revisit_gradient_v31/pipeline_status.json').read_text())
    proof_config=json.loads((r/'outer_revisit_gradient_v31/config.json').read_text())
    if proof['phase']!='passed' or not proof['zero_intervention_latent_exact'] or proof['base_parameter_gradients']:
        raise ValueError('Require actual outer full-terminal zero/gradient proof')
    for name in ('runtime.py','outer_shared_revisit.py'):
        if proof_config['code_sha256'][name]!=file_hash(r/name):raise ValueError('Outer interface changed after proof')
    report=json.loads((teacher/'paired_report.json').read_text())
    records=list(map(json.loads,(teacher/'all_actions.jsonl').read_text().splitlines()))
    gate=validate_training_pairs(report,records)
    verification=json.loads((r/'v23_portable_verification.json').read_text())
    if not verification['portable_exact'] or verification['paired_report_sha256']!=file_hash(teacher/'paired_report.json'):
        raise ValueError('Teacher source differs from portable verification')
    names=('run_outer_capacity_pilot.py','train_outer_operator.py','build_outer_trainfit_candidates.py','outer_shared_revisit.py',
           'runtime.py','evidence_encoder.py','training_pairs.py','candidate_provenance.py')
    freeze_config(out/'config.json',dict(training_gate=gate,steps=50,lr=1e-4,width=128,window=4,level=2,
        outer_shared_core_indices=[26,27],initialization_seed=20261005,state_sampling_seed=20261004,
        preservation_every=3,checkpoint_selection='fixed final step50',val_images=0,controller=False,
        trainfit_images=10,source_sha256={n:file_hash(r/n) for n in names},
        comparison='Outer capacity diagnostic; differs from inner dimensionality, not an equal-FLOPs causal comparison',
        dry_run_sha256=file_hash(r/'outer_revisit_gradient_v31/pipeline_status.json')))
    (out/'source').mkdir()
    for n in names:shutil.copy2(r/n,out/'source'/n)
    update('training',optimizer_updates=0)
    run([py,str(r/'train_outer_operator.py'),'--work',str(w),'--states',str(states),
         '--records',str(teacher/'all_actions.jsonl'),'--paired-report',str(teacher/'paired_report.json'),
         '--output',str(out/'outer'),'--steps','50','--lr','1e-4','--level','2','--width','128','--window','4'],'train')
    update('trainfit_generation',optimizer_updates=50)
    run([py,str(r/'build_outer_trainfit_candidates.py'),'--work',str(w),'--states',str(r/'states/trainfit_v27'),
         '--evidence-states',str(states),'--output',str(out/'trainfit'),
         '--checkpoint',str(out/'outer/operator_step50.pt')],'generation')
    update('trainfit_evaluation',optimizer_updates=50)
    official=r/'evaluator_runtime_v1/venv'
    command=[py,str(r/'evaluate_oracle_pipeline_available.py'),'--work',str(w),'--experiment','outer_capacity_v33/trainfit',
        '--wait-generation','--official-python',str(official/'bin/python'),'--official-model-config',
        str(official/'lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'),
        '--official-model-path',str(w/'stage2b/geneval_models'),'--evaluator-verification',
        str(r/'minimal_teacher_geometry_dev6_v12/evaluation_recovery_v1/compatibility_report.json')]
    with (out/'evaluation.log').open('w') as stream:subprocess.run(command,cwd=w,stdout=stream,stderr=subprocess.STDOUT,check=True)
    from export_results import export
    export(out/'trainfit')
    update('complete',optimizer_updates=50,trainfit_images=10,val_images=0,
        next='Assess TRAIN semantic/quality/retention and evidence controls; no automatic val or controller')
except Exception as error:update('failed',error=str(error));raise
