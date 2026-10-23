"""Matched initialization, TRAIN-only spatial capacity pilot. No controller."""
import concurrent.futures,json,os,pathlib,shutil,subprocess,time
from launch_grounding_diagnostic import GPUS
from candidate_provenance import file_hash,freeze_config
from training_pairs import validate_training_pairs

w=pathlib.Path('/home/shy_23/iterative_visual_stage1_20261003');r=w/'stage3_grounded_repair'
out=r/'spatial_capacity_v28';out.mkdir()
python='/home/shy_23/venvs/show_o2/bin/python';status=out/'pipeline_status.json'
teacher=r/'oracle_encoded_correlation_train24x2_v23';states=r/'states/train_pilot_minimal_v14'
def update(phase,**kw):status.write_text(json.dumps(dict(phase=phase,time=time.time(),controller_started=False,**kw),indent=2))
def env(gpu):
    e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=GPUS[gpu],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TORCHDYNAMO_DISABLE='1',PYTHONPATH=str(w/'deps'));return e
def run(cmd,gpu,log,label):
    deadline=time.time()+1800
    while int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())>=2000:
        if time.time()>deadline:raise TimeoutError('GPU occupied; no interruption of others')
        time.sleep(20)
    with log.open('w') as stream:
        p=subprocess.Popen(cmd,cwd=w,env=env(gpu),stdout=stream,stderr=subprocess.STDOUT)
        (out/(label+'.json')).write_text(json.dumps(dict(pid=p.pid,gpu=gpu,status='running')))
        if p.wait(timeout=1800):raise RuntimeError(label+' failed: '+str(log))
    (out/(label+'.json')).write_text(json.dumps(dict(pid=p.pid,gpu=gpu,status='complete')))
try:
    gate=validate_training_pairs(json.loads((teacher/'paired_report.json').read_text()),
        list(map(json.loads,(teacher/'all_actions.jsonl').read_text().splitlines())))
    verification=json.loads((r/'v23_portable_verification.json').read_text())
    if not verification['portable_exact'] or verification['paired_report_sha256']!=file_hash(teacher/'paired_report.json'):
        raise ValueError('Teacher portable verification mismatch')
    if json.loads((r/'token_trainfit_v27/pipeline_status.json').read_text())['phase']!='complete':
        raise ValueError('Require completed TRAIN fit diagnosis before changing capacity')
    names=('run_spatial_capacity_pilot.py','spatial_token_revisit.py','test_spatial_token_revisit.py',
        'train_capacity_matched_operator.py','check_spatial_rollout_gradient.py','build_spatial_trainfit_candidates.py',
        'build_token_trainfit_candidates.py','token_conditioned_revisit.py','evidence_conditioned_revisit.py',
        'evidence_encoder.py','runtime.py','training_pairs.py','candidate_provenance.py')
    freeze_config(out/'config.json',dict(training_gate=gate,teacher_report_sha256=file_hash(teacher/'paired_report.json'),
        steps_per_arm=50,lr=1e-4,width=128,window=4,level=2,shared_core_index=8,
        initialization_seed=20261005,state_sampling_seed=20261004,arms=['film','spatial'],
        preservation_every=3,checkpoint_selection='fixed final step50',
        rationale='v27 cannot fit TRAIN teacher; nonzero hidden changes exclude total BF16 inactivity',
        diagnostic_train_states=2,trainfit_terminal_images=16,controller=False,val_images=0,
        source_sha256={n:file_hash(r/n) for n in names}))
    (out/'source').mkdir()
    for n in names:shutil.copy2(r/n,out/'source'/n)
    update('unit_tests')
    with (out/'unit_tests.log').open('w') as log:
        subprocess.run([python,'-m','unittest','test_spatial_token_revisit.py'],cwd=r,stdout=log,stderr=subprocess.STDOUT,check=True)
    update('real_gradient_dry_run')
    state=next(p for p in states.rglob('*_i35.pt') if p.stem=='train_extra_00021_seed101_i35')
    run([python,str(r/'check_spatial_rollout_gradient.py'),'--work',str(w),'--state',str(state),
         '--output',str(out/'spatial_gradient.json'),'--core-index','8'],0,out/'gradient.log','gradient')
    update('training',steps_per_arm=50)
    def train(arm,gpu):
        run([python,str(r/'train_capacity_matched_operator.py'),'--work',str(w),'--states',str(states),
            '--records',str(teacher/'all_actions.jsonl'),'--paired-report',str(teacher/'paired_report.json'),
            '--output',str(out/arm),'--steps','50','--lr','1e-4','--level','2','--width','128',
            '--window','4','--core-index','8','--operator-kind','spatial_tokenwise' if arm=='spatial' else 'tokenwise'],
            gpu,out/(arm+'_train.log'),arm+'_train')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(train,'film',4),pool.submit(train,'spatial',6)]
        for job in jobs:job.result()
    f=json.loads((out/'film/training_config.json').read_text())['adapter_initialization_hashes']
    s=json.loads((out/'spatial/training_config.json').read_text())['adapter_initialization_hashes']
    if {k:s[k] for k in f}!=f:raise ValueError('Common adapter initialization is not matched')
    (out/'matched_initialization.json').write_text(json.dumps(dict(passed=True,seed=20261005,common_weights=f),indent=2))
    update('trainfit_generation',optimizer_updates=100)
    def fit(arm,gpu):
        directory=out/(arm+'_trainfit');directory.mkdir()
        builder='build_spatial_trainfit_candidates.py' if arm=='spatial' else 'build_token_trainfit_candidates.py'
        run([python,str(r/builder),'--work',str(w),'--states',str(r/'states/trainfit_v27'),'--output',str(directory),
            '--checkpoint',str(out/arm/'operator_step50.pt'),'--core-index','8','--level','2','--window','4',
            '--actions','baseline,conditional_zero,conditional_revisit,opposite_evidence'],gpu,
            directory/'generation.log',arm+'_fit')
        (directory/'generation_status.json').write_text(json.dumps(dict(success=True,expected_images=8)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(fit,'film',4),pool.submit(fit,'spatial',6)]
        for job in jobs:job.result()
    for arm in ('film','spatial'):
        update('trainfit_evaluation',arm=arm,optimizer_updates=100)
        official=r/'evaluator_runtime_v1/venv';directory=out/(arm+'_trainfit')
        cmd=[python,str(r/'evaluate_oracle_pipeline_available.py'),'--work',str(w),'--experiment',f'spatial_capacity_v28/{arm}_trainfit',
            '--wait-generation','--official-python',str(official/'bin/python'),'--official-model-config',
            str(official/'lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'),
            '--official-model-path',str(w/'stage2b/geneval_models'),'--evaluator-verification',
            str(r/'minimal_teacher_geometry_dev6_v12/evaluation_recovery_v1/compatibility_report.json')]
        with (directory/'evaluation.log').open('w') as log:subprocess.run(cmd,cwd=w,stdout=log,stderr=subprocess.STDOUT,check=True)
        from export_results import export
        export(directory)
    update('complete',optimizer_updates=100,trainfit_terminal_images=16,val_images=0,
        next='Assess train capacity and preservation; only then independent val, no controller')
except Exception as error:update('failed',error=str(error));raise
