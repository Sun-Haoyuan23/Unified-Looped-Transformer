"""Preserve active models; move only the finite waiting launcher to idle scheduling."""
import json,os,shutil,signal,subprocess,time
from pathlib import Path
from candidate_provenance import file_hash
from launch_grounding_diagnostic import GPUS,write
from recurrent_training_protocol import STUDIES

w=Path('/home/shy_23/iterative_visual_stage1_20261003');r=w/'stage3_grounded_repair'
out=r/'recurrent_train_v46';recovery=out/'resource_recovery_v1';py='/home/shy_23/venvs/show_o2/bin/python'
def process(pid):
 p=Path('/proc')/str(pid)
 if not p.exists():return None
 if p.stat().st_uid!=os.getuid():raise ValueError('Process is not owned by the authorized account')
 command=(p/'cmdline').read_bytes().decode().split('\0')
 state=(p/'stat').read_text().split(') ',1)[1].split()[0]
 return None if state=='Z' else command
def update(phase,**kw):
 state=dict(phase=phase,time=time.time(),controller_started=False,resource_recovery=str(recovery),**kw)
 write(recovery/'pipeline_status.json',state);write(out/'pipeline_status.json',state)

def main():
 recovery.mkdir()
 try:
  parent=14165;cmd=process(parent)
  if cmd is None or str(r/'run_recurrent_training.py') not in cmd:raise ValueError('Original waiting launcher identity changed')
  states={a:json.loads((out/(a+'.json')).read_text()) for a in STUDIES}
  if any(states[a]['status']!='waiting_for_gpu' or states[a].get('pid') for a in ('outer_ilsd','inner_ilsd')):
   raise ValueError('Pending studies are no longer empty waiting workers')
  running={}
  for a in ('joint_ilsd','joint_terminal'):
   pid=states[a]['pid'];command=process(pid)
   if command is None or str(r/'train_recurrent_operator.py') not in command or '--study' not in command or command[command.index('--study')+1]!=a:
    raise ValueError('Active model identity changed; do not interrupt it')
   running[a]=dict(pid=pid,gpu=states[a]['gpu'])
  originals={n:file_hash(out/'source'/n) for n in ('train_recurrent_operator.py','trainable_recurrent_state.py','recurrent_training_protocol.py','state_space_revisit.py','runtime.py')}
  if any(file_hash(r/n)!=digest for n,digest in originals.items()):raise ValueError('Frozen training source changed')
  record=dict(original_launcher_pid=parent,original_workers=states,active_models_preserved=running,
   original_configuration_sha256=file_hash(out/'config.json'),frozen_source_sha256=originals,
   reason='Other-user tasks newly occupyGPU0/4; avoid30min fixed-GPU timeout and preserve activeGPU6/7 models',
   action='SIGTERM only own CPU waiting launcher; never its model children or other users',time=time.time())
  write(recovery/'resource_reschedule.json',record)
  (recovery/'source').mkdir()
  for n in ('resume_recurrent_resources.py','run_recurrent_trainfit_resources.py','evaluate_oracle_pipeline_two_available.py'):
   shutil.copy2(r/n,recovery/'source'/n)
  os.kill(parent,signal.SIGTERM)
  pending=['outer_ilsd','inner_ilsd'];done=set();deadline=time.time()+6*3600
  update('training',active_model_pids=running,pending_studies=pending)
  while pending or running:
   if time.time()>deadline:raise TimeoutError('No resource progress before6hour deadline; models not killed')
   for a,info in list(running.items()):
    if (out/a/'training_complete.json').exists():
     done.add(a);running.pop(a)
     write(out/(a+'.json'),dict(study=a,gpu=info['gpu'],pid=info['pid'],status='complete',resource_recovery=True))
    elif process(info['pid']) is None:raise RuntimeError(a+' model ended without completed fixed training')
   reserved={v['gpu'] for v in running.values()}
   for a in list(pending):
    idle=[]
    for gpu in (0,4,6,7):
     if gpu in reserved:continue
     used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
     if used<2000:idle.append(gpu)
    if not idle:break
    gpu=idle[0]
    if (out/a).exists():raise ValueError('Pending training output unexpectedly exists; no duplicate training')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCHDYNAMO_DISABLE='1',
      OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(w/'deps'),PYTHONWARNINGS='ignore')
    command=[py,str(r/'train_recurrent_operator.py'),'--work',str(w),'--states',str(r/'states/repair_train_targets_v36'),
      '--records',str(r/'repair_train_targets_v36/all_actions.jsonl'),'--paired-report',str(r/'repair_train_targets_v36/paired_report.json'),
      '--output',str(out/a),'--steps','100','--width','64','--window','1','--study',a]
    with (out/(a+'.log')).open('w') as log:q=subprocess.Popen(command,cwd=w,env=env,stdout=log,stderr=subprocess.STDOUT)
    running[a]=dict(pid=q.pid,gpu=gpu);reserved.add(gpu);pending.remove(a)
    write(out/(a+'.json'),dict(study=a,gpu=gpu,pid=q.pid,status='training',resource_recovery=True))
   update('training',active_model_pids=running,pending_studies=pending,completed_studies=sorted(done))
   time.sleep(10)
  initial={};sequences={}
  for a in STUDIES:
   c=json.loads((out/a/'training_config.json').read_text());initial[a]=c['adapter_initialization_hashes']
   rows=list(map(json.loads,(out/a/'train.jsonl').read_text().splitlines()))
   if [q['step'] for q in rows]!=list(range(1,101)) or not any(q['low_rank_gradient_abs_sum']>0 for q in rows):raise ValueError('Incomplete/nonlearning fixed training')
   sequences[a]=[(q['state_id'],q['preservation_step']) for q in rows]
  if initial['joint_ilsd']!=initial['joint_terminal'] or any(q!=sequences['joint_ilsd'] for q in sequences.values()):raise ValueError('Objective control init or sample mismatch')
  for a in ('outer_ilsd','inner_ilsd'):
   if any(initial['joint_ilsd'][k]!=v for k,v in initial[a].items()):raise ValueError('Branch init mismatch')
  if any(file_hash(r/n)!=h for n,h in originals.items()) or file_hash(out/'config.json')!=record['original_configuration_sha256']:raise ValueError('Frozen source/config changed')
  write(out/'matched_initialization.json',dict(passed=True,all_common_and_branch_tensors_matched=True,state_sampling_exact=True,resource_recovery=True))
  update('training_complete',optimizer_updates=400,val_images=0)
  with (recovery/'trainfit_pipeline.log').open('w') as log:
   subprocess.run([py,str(r/'run_recurrent_trainfit_resources.py')],cwd=w,stdout=log,stderr=subprocess.STDOUT,check=True)
  update('complete',optimizer_updates=400,trainfit_images=144,val_images=0)
 except Exception as error:update('failed',error=str(error));raise

if __name__=='__main__':main()
