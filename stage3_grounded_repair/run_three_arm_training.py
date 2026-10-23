"""Finite TRAIN capacity comparison; no automatic val/controller."""
import concurrent.futures,json,os,shutil,subprocess,time
from pathlib import Path
from candidate_provenance import file_hash,freeze_config
from training_pairs import validate_training_pairs
from launch_grounding_diagnostic import GPUS,write

w=Path('/home/shy_23/iterative_visual_stage1_20261003');r=w/'stage3_grounded_repair'
out=r/'two_level_train_v38';out.mkdir();teacher=r/'repair_train_targets_v36';py='/home/shy_23/venvs/show_o2/bin/python'

def update(phase,**kw):write(out/'pipeline_status.json',dict(phase=phase,time=time.time(),controller_started=False,**kw))

try:
    proof=json.loads((r/'two_level_gradient_v37/pipeline_status.json').read_text())
    if proof['phase']!='passed' or not proof['zero_intervention_latent_exact'] or proof['base_parameter_gradients']:
        raise ValueError('Require actual joint full-terminal gradient/zero proof')
    gate=validate_training_pairs(json.loads((teacher/'paired_report.json').read_text()),
                                 list(map(json.loads,(teacher/'all_actions.jsonl').read_text().splitlines())))
    if gate['selected_teacher_states']!=3 or gate['preservation_states']!=35:raise ValueError('Strict training cohort changed')
    names=('run_three_arm_training.py','train_two_level_operator.py','two_level_operator.py','two_level_replay.py','check_shared_memory_gradient.py','test_two_level_operator.py',
           'outer_shared_revisit.py','token_conditioned_revisit.py','evidence_conditioned_revisit.py',
           'runtime.py','evidence_encoder.py','training_pairs.py')
    freeze_config(out/'config.json',dict(training_gate=gate,steps_per_arm=100,lr=1e-4,width=128,level=2,window=4,
        teacher_report_sha256=file_hash(teacher/'paired_report.json'),teacher_records_sha256=file_hash(teacher/'all_actions.jsonl'),
        source_sha256={n:file_hash(r/n) for n in names},common_initialization_seed=20261005,state_sampling_seed=20261004,
        preservation_every=3,checkpoint_selection='fixed step100',val_images=0,controller=False,
        rationale='New qualified small-object teachers have20x smaller endpoint displacement than global mirror; balance relative improvement',
        comparison='Capacity diagnostic with same data/objective/common initialization; different costs, not equal-FLOPs comparison'))
    (out/'source').mkdir()
    for n in names:shutil.copy2(r/n,out/'source'/n)
    update('unit_and_shared_memory_gradient')
    with (out/'unit_tests.log').open('w') as log:
        subprocess.run([py,'-m','unittest','test_two_level_operator.py'],cwd=r,stdout=log,stderr=subprocess.STDOUT,check=True)
    if int(subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())>=2000:
        raise ValueError('Proof GPU7 occupied; do not interrupt others')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPUS[7],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCHDYNAMO_DISABLE='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(w/'deps'),PYTHONWARNINGS='ignore')
    with (out/'shared_memory_gradient.log').open('w') as log:
        subprocess.run([py,str(r/'check_shared_memory_gradient.py')],cwd=w,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    if json.loads((r/'shared_memory_gradient_v38/pipeline_status.json').read_text())['phase']!='passed':raise ValueError('Shared memory gradient proof failed')
    update('training',steps_per_arm=100)
    def worker(arm,gpu):
        state=dict(arm=arm,gpu=gpu,status='waiting_for_gpu');path=out/(arm+'.json');write(path,state)
        try:
            deadline=time.time()+1800
            while int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())>=2000:
                if time.time()>deadline:raise TimeoutError('GPU occupied; no other process interrupted')
                time.sleep(20)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                TORCHDYNAMO_DISABLE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONWARNINGS='ignore',PYTHONPATH=str(w/'deps'))
            command=[py,str(r/'train_two_level_operator.py'),'--work',str(w),'--states',str(r/'states/repair_train_targets_v36'),
                '--records',str(teacher/'all_actions.jsonl'),'--paired-report',str(teacher/'paired_report.json'),
                '--output',str(out/arm),'--steps','100','--lr','1e-4','--level','2','--width','128','--window','4','--arm',arm]
            with (out/(arm+'.log')).open('w') as stream:
                p=subprocess.Popen(command,cwd=w,env=env,stdout=stream,stderr=subprocess.STDOUT)
                state.update(pid=p.pid,status='training');write(path,state)
                if p.wait(timeout=3600):raise RuntimeError(arm+' training failed')
            state.update(status='complete')
        except Exception as error:state.update(status='failed',error=str(error))
        write(path,state);return state
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        states=list(pool.map(lambda item:worker(*item),(('outer',0),('inner',4),('joint',6))))
    if any(q['status']!='complete' for q in states):raise ValueError('One or more training arms failed')
    initial=[json.loads((out/arm/'training_config.json').read_text())['common_evidence_initialization_hashes'] for arm in ('outer','inner','joint')]
    if not initial[0]==initial[1]==initial[2]:raise ValueError('Common evidence initialization mismatch')
    write(out/'matched_initialization.json',dict(passed=True,common_hashes=initial[0]))
    update('training_complete',optimizer_updates=300,val_images=0,
        next='Fixed step100 TRAIN terminal diagnosis for all3 teachers and real correct states with zero/shuffled/opposite evidence; no val/controller')
except Exception as error:update('failed',error=str(error));raise
