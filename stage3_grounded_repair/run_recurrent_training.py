"""Four fixed studies train shared recurrent attention and terminal ILSD."""
import concurrent.futures,json,os,shutil,subprocess,time
from pathlib import Path
from candidate_provenance import file_hash,freeze_config
from training_pairs import validate_training_pairs
from launch_grounding_diagnostic import GPUS,write
from recurrent_training_protocol import STUDIES

w=Path('/home/shy_23/iterative_visual_stage1_20261003');r=w/'stage3_grounded_repair'
out=r/'recurrent_train_v46';py='/home/shy_23/venvs/show_o2/bin/python'
def update(phase,**kw):write(out/'pipeline_status.json',dict(phase=phase,time=time.time(),controller_started=False,**kw))

def main():
 out.mkdir()
 try:
  proof=r/'recurrent_core_gradient_v46'
  q=json.loads((proof/'pipeline_status.json').read_text())
  if (q['phase']!='passed' or q['zero_terminal_budgets_exact']!=[1,2] or q['optimizer_updates']!=0 or
      q['base_parameter_gradients'] or not q['shared_base_cores_exact'] or min(q['low_rank_gradient_abs_sum'].values())<=0):
   raise ValueError('Require actual nested terminal/core learning proof')
  for name,digest in json.loads((proof/'config.json').read_text())['source_sha256'].items():
   if file_hash(r/name)!=digest:raise ValueError('Verified core interface changed: '+name)
  teacher=r/'repair_train_targets_v36';gate=validate_training_pairs(json.loads((teacher/'paired_report.json').read_text()),list(map(json.loads,(teacher/'all_actions.jsonl').read_text().splitlines())))
  if gate['selected_teacher_states']!=3 or gate['preservation_states']!=35:raise ValueError('Strict TRAIN gate changed')
  names=('run_recurrent_training.py','train_recurrent_operator.py','trainable_recurrent_state.py','recurrent_training_protocol.py',
   'test_trainable_recurrent_state.py','check_recurrent_core_gradient.py','state_space_revisit.py','runtime.py','evidence_encoder.py','training_pairs.py')
  freeze_config(out/'config.json',dict(training_gate=gate,steps_per_study=100,width=64,level=2,window=1,rank=4,max_recurrent_loops=2,
   lr=1e-4,source_sha256={n:file_hash(r/n) for n in names},gradient_proof_sha256=file_hash(proof/'pipeline_status.json'),
   teacher_report_sha256=file_hash(teacher/'paired_report.json'),teacher_records_sha256=file_hash(teacher/'all_actions.jsonl'),
   common_initialization_seed=20261005,state_sampling_seed=20261004,preservation_every=3,
   checkpoint_selection='fixed final step100 before terminal diagnosis',controller=False,val_images=0,
   rationale='v45 frozen shared groups and latent readout had zero teacher flips and one preservation harm; train the recurrent transformation itself and supervise exact nested loop prefixes',
   reference='https://arxiv.org/html/2604.09168v3 section4; adapted terminal ILSD, not original denoising training',
   direct_control='joint_ilsd vs joint_terminal same initialization/model/data/maxloops/updates, objective differs; costs measured separately',
   comparison='outer/inner/joint retained; each frozen checkpoint evaluated at budgets1/2, not an equal-training-FLOPs claim'))
  (out/'source').mkdir()
  for n in names:shutil.copy2(r/n,out/'source'/n)
  with (out/'unit_tests.log').open('w') as log:subprocess.run([py,'-m','unittest','test_trainable_recurrent_state.py'],cwd=r,stdout=log,stderr=subprocess.STDOUT,check=True)
  update('training',steps_per_study=100)
  def worker(study,gpu):
   state=dict(study=study,gpu=gpu,status='waiting_for_gpu');path=out/(study+'.json');write(path,state)
   try:
    deadline=time.time()+1800
    while int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())>=2000:
     if time.time()>deadline:raise TimeoutError('GPU occupied; no other task interrupted')
     time.sleep(20)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCHDYNAMO_DISABLE='1',
     OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(w/'deps'),PYTHONWARNINGS='ignore')
    command=[py,str(r/'train_recurrent_operator.py'),'--work',str(w),'--states',str(r/'states/repair_train_targets_v36'),
     '--records',str(teacher/'all_actions.jsonl'),'--paired-report',str(teacher/'paired_report.json'),'--output',str(out/study),
     '--steps','100','--width','64','--window','1','--study',study]
    with (out/(study+'.log')).open('w') as log:
     p=subprocess.Popen(command,cwd=w,env=env,stdout=log,stderr=subprocess.STDOUT);state.update(pid=p.pid,status='training');write(path,state)
     if p.wait(timeout=5400):raise RuntimeError(study+' training failed')
    state.update(status='complete')
   except Exception as error:state.update(status='failed',error=str(error))
   write(path,state);return state
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:states=list(pool.map(lambda q:worker(*q),zip(STUDIES,(0,4,6,7))))
  if any(q['status']!='complete' for q in states):raise ValueError('One or more recurrent studies failed')
  configs={a:json.loads((out/a/'training_config.json').read_text()) for a in STUDIES}
  initial={a:c['adapter_initialization_hashes'] for a,c in configs.items()}
  if initial['joint_ilsd']!=initial['joint_terminal']:raise ValueError('Objective control initialization mismatch')
  for study in ('outer_ilsd','inner_ilsd'):
   if any(initial['joint_ilsd'][k]!=v for k,v in initial[study].items()):raise ValueError('Shared branch initialization mismatch')
  sequences={a:[(q['state_id'],q['preservation_step']) for q in map(json.loads,(out/a/'train.jsonl').read_text().splitlines())] for a in STUDIES}
  if any(len(q)!=100 for q in sequences.values()) or any(q!=sequences['joint_ilsd'] for q in sequences.values()):raise ValueError('Sampling/update sequence mismatch')
  for study in STUDIES:
   if not any(q['low_rank_gradient_abs_sum']>0 for q in map(json.loads,(out/study/'train.jsonl').read_text().splitlines())):
    raise ValueError('No actual recurrent core gradient during training')
  write(out/'matched_initialization.json',dict(passed=True,all_common_and_branch_tensors_matched=True,state_sampling_exact=True))
  update('training_complete',optimizer_updates=400,val_images=0)
  with (out/'trainfit_pipeline.log').open('w') as log:subprocess.run([py,str(r/'run_recurrent_trainfit.py')],cwd=w,stdout=log,stderr=subprocess.STDOUT,check=True)
  update('complete',optimizer_updates=400,trainfit_images=144,val_images=0,next='Check semantics/quality/ILSD control/loop1vs2 before independent val')
 except Exception as error:update('failed',error=str(error));raise

if __name__=='__main__':main()
