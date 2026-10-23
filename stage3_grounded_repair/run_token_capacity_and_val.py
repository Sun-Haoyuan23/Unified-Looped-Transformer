"""Quality-gated shared-loop capacity pilot, then independent fixed-checkpoint val.

50 optimizer steps per core, identical state sampling/hyperparameters. A single
successful TRAIN teacher is an overfit/capacity pilot, never broad evidence.
Validation is frozen before training; controller training is not implemented.
"""
import concurrent.futures,json,os,subprocess,time,shutil
from pathlib import Path
from collections import defaultdict
from candidate_provenance import file_hash,freeze_config
from training_pairs import validate_training_pairs
from launch_grounding_diagnostic import GPUS

work=Path('/home/shy_23/iterative_visual_stage1_20261003');root=work/'stage3_grounded_repair'
python='/home/shy_23/venvs/show_o2/bin/python'
teacher=root/'oracle_encoded_correlation_train24x2_v23'
run=root/'token_capacity_and_val_v25';run.mkdir()
status_path=run/'pipeline_status.json'

def update(phase,**kwargs):
    status_path.write_text(json.dumps(dict(phase=phase,time=time.time(),controller_started=False,**kwargs),indent=2))

def env(gpu):
    e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=GPUS[gpu],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TORCHDYNAMO_DISABLE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONWARNINGS='ignore',PYTHONPATH=str(work/'deps'))
    return e

def execute(cmd,gpu,log,status):
    path=run/(status+'.json')
    deadline=time.time()+1800
    while int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())>=2000:
        if time.time()>deadline:raise TimeoutError('GPU occupied; never interrupt another user')
        time.sleep(20)
    with log.open('w') as stream:
        p=subprocess.Popen(cmd,cwd=work,env=env(gpu),stdout=stream,stderr=subprocess.STDOUT)
        path.write_text(json.dumps(dict(status='running',pid=p.pid,gpu=gpu,started=time.time())))
        code=p.wait(timeout=2400)
    if code:raise RuntimeError(f'{status} failed; inspect {log}')
    path.write_text(json.dumps(dict(status='complete',pid=p.pid,gpu=gpu,completed=time.time())))

try:
    report=json.loads((teacher/'paired_report.json').read_text())
    records=list(map(json.loads,(teacher/'all_actions.jsonl').read_text().splitlines()))
    gate=validate_training_pairs(report,records)
    verified=json.loads((root/'v23_portable_verification.json').read_text())
    if verified.get('portable_exact') is not True or verified['paired_report_sha256']!=file_hash(teacher/'paired_report.json'):
        raise ValueError('Require exact independent portable reanalysis before training')
    if json.loads((root/'token_rollout_core8_v20.json').read_text())['status']!='passed':raise ValueError('Token real-model gradient check incomplete')
    # No images/scores/terminal labels enter val prompt selection.
    source=root/'splits_v1/val.jsonl';rows=list(map(json.loads,source.read_text().splitlines()));buckets=defaultdict(list)
    for row in rows:
        if row['split']!='val' or row['metadata']['prompt']!=row['prompt']:raise ValueError('Non-val source')
        buckets[row['metadata']['include'][1]['position'][0]].append(row)
    selected=[]
    for relation in ('left of','right of','above','below'):
        group=sorted(buckets[relation],key=lambda r:r['prompt_id'])
        if len(group)<3:raise ValueError('Insufficient balanced val source')
        selected.extend(dict(r,seeds=[101,202]) for r in group[:3])
    selected.sort(key=lambda r:r['prompt_id'])
    if len({r['prompt'] for r in selected})!=12:raise ValueError('Duplicate val prompts')
    val_tasks=run/'val_prompts.json';val_tasks.write_text(json.dumps(selected,indent=2))
    names=('run_token_capacity_and_val.py','train_token_repair_operator.py','token_conditioned_revisit.py',
        'evidence_conditioned_revisit.py','evidence_encoder.py','runtime.py','training_pairs.py',
        'build_token_validation_candidates.py','capture_repair_states.py','candidate_provenance.py',
        'evaluate_oracle_pipeline.py','evaluate_oracle_pipeline_available.py','export_results.py')
    freeze_config(run/'config.json',dict(training_gate=gate,teacher_report_sha256=file_hash(teacher/'paired_report.json'),
        portable_verification_sha256=file_hash(root/'v23_portable_verification.json'),val_source_sha256=file_hash(source),
        val_tasks_sha256=file_hash(val_tasks),val_prompts=12,val_states=24,steps=50,lr=1e-4,window=4,level=2,width=128,
        core_indices=[8,2],checkpoint_selection='fixed final step50 before val',controller=False,
        source_sha256={n:file_hash(root/n) for n in names},scope='single-teacher capacity pilot, independent val, not confirmatory benchmark'))
    (run/'source').mkdir()
    for n in names:shutil.copy2(root/n,run/'source'/n)
    (run/'source_snapshot.json').write_text(json.dumps({n:file_hash(run/'source'/n) for n in names},indent=2))
    update('training_and_val_capture',training_gate=gate,steps_per_core=50)
    def train(core,gpu):
        output=run/f'core{core}'
        cmd=[python,str(root/'train_token_repair_operator.py'),'--work',str(work),'--states',str(root/'states/train_pilot_minimal_v14'),
            '--records',str(teacher/'all_actions.jsonl'),'--paired-report',str(teacher/'paired_report.json'),'--output',str(output),
            '--steps','50','--lr','1e-4','--level','2','--width','128','--window','4','--core-index',str(core)]
        execute(cmd,gpu,run/f'train_core{core}.log',f'train_core{core}')
    def capture(shard,gpu):
        cmd=[python,str(root/'capture_repair_states.py'),'--work',str(work),'--tasks',str(val_tasks),
             '--output',str(root/f'states/val_token_v25/shard{shard}'),'--shard',str(shard),'--num-shards','2','--indices','35','--split','val']
        execute(cmd,gpu,run/f'capture_val_shard{shard}.log',f'capture_val_shard{shard}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(train,8,4),pool.submit(train,2,6),pool.submit(capture,0,0),pool.submit(capture,1,7)]
        for job in jobs:job.result()
    states=list((root/'states/val_token_v25').rglob('*_i35.pt'))
    if len(states)!=24:raise ValueError('Missing independent val captures')
    update('validation_generation',checkpoint_step=50,val_states=24)
    actions='baseline,conditional_zero,conditional_revisit,shuffled_evidence,opposite_evidence'
    def generate(core,shard,gpu):
        output=root/f'token_repair_val_v26/core{core}/shard{shard}';output.mkdir(parents=True)
        cmd=[python,str(root/'build_token_validation_candidates.py'),'--work',str(work),'--states',str(root/'states/val_token_v25'),
            '--output',str(output),'--actions',actions,'--checkpoint',str(run/f'core{core}/operator_step50.pt'),
            '--level','2','--core-index',str(core),'--window','4','--shard',str(shard),'--num-shards','2']
        execute(cmd,gpu,run/f'val_core{core}_shard{shard}.log',f'val_core{core}_shard{shard}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(generate,8,0,0),pool.submit(generate,8,1,4),pool.submit(generate,2,0,6),pool.submit(generate,2,1,7)]
        for job in jobs:job.result()
    for core in (8,2):
        out=root/f'token_repair_val_v26/core{core}'
        (out/'generation_status.json').write_text(json.dumps(dict(success=True,expected_images=120,checkpoint_step=50)))
        update('validation_evaluation',core=core)
        official=root/'evaluator_runtime_v1/venv'
        cmd=[python,str(root/'evaluate_oracle_pipeline_available.py'),'--work',str(work),'--experiment',f'token_repair_val_v26/core{core}',
            '--wait-generation','--official-python',str(official/'bin/python'),'--official-model-config',
            str(official/'lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'),
            '--official-model-path',str(work/'stage2b/geneval_models'),'--evaluator-verification',
            str(root/'minimal_teacher_geometry_dev6_v12/evaluation_recovery_v1/compatibility_report.json')]
        with (out/'evaluation_launcher.log').open('w') as log:subprocess.run(cmd,cwd=work,stdout=log,stderr=subprocess.STDOUT,check=True)
        from export_results import export
        export(out)
    update('complete',optimizer_steps=100,validation_images=240,controller_started_forbidden=True,
           next='Analyze core8/core2 and shuffled/opposite val effects; expand TRAIN teachers before broader claims')
except Exception as error:
    update('failed',error=str(error));raise
