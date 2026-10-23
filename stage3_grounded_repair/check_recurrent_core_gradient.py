"""Actual nested-terminal low-rank shared-core proof; no optimizer update."""
from pathlib import Path
import json,shutil,time
import torch
from runtime import Runtime,tensor_hash,write_json
from evidence_encoder import condition_tensor
from training_pairs import selected_pairs,validate_training_pairs
from candidate_provenance import file_hash
from trainable_recurrent_state import make_operator,capture_features,full_model_context,replay_delta

w=Path('/home/shy_23/iterative_visual_stage1_20261003');r=w/'stage3_grounded_repair';out=r/'recurrent_core_gradient_v46';out.mkdir()
def update(phase,**kw):write_json(out/'pipeline_status.json',dict(phase=phase,time=time.time(),optimizer_updates=0,**kw))
try:
 names=('check_recurrent_core_gradient.py','trainable_recurrent_state.py','state_space_revisit.py','runtime.py','evidence_encoder.py','training_pairs.py')
 (out/'source').mkdir()
 for n in names:shutil.copy2(r/n,out/'source'/n)
 teacher=r/'repair_train_targets_v36';report=json.loads((teacher/'paired_report.json').read_text());records=list(map(json.loads,(teacher/'all_actions.jsonl').read_text().splitlines()))
 gate=validate_training_pairs(report,records);pairs=selected_pairs(report,records)
 if gate['selected_teacher_states']!=3 or gate['preservation_states']!=35:raise ValueError('Strict TRAIN gate changed')
 sid='train_extra_00021_seed101_i35';path=r/'states/repair_train_targets_v36'/f'{sid}.pt'
 update('loading');rt=Runtime(w,vae=False);base=list(rt.model.parameters());state=torch.load(path,map_location='cpu',weights_only=False)
 if state['split']!='train' or state['flow_index']!=35 or not state['zero_intervention_test']['passed']:raise ValueError('Invalid TRAIN state')
 torch.manual_seed(20261005);op=make_operator(rt);op.condition=condition_tensor(state['evidence'],state['true_t'],rt.device)
 hashes=op.core_hashes();row=pairs[sid]
 if file_hash(row['final_latent'])!=row['final_latent_sha256']:raise ValueError('Changed teacher')
 target=torch.load(row['final_latent'],map_location=rt.device,weights_only=False)['latent'].float()
 baseline=state['baseline_final_latent'].to(rt.device).float()
 write_json(out/'config.json',dict(source_sha256={n:file_hash(r/n) for n in names},state_sha256=file_hash(path),training_gate=gate,
   rank=4,budgets=[1,2],readout_probe_std=.001,updates=0,base_weights_referenced_not_copied=True))
 with full_model_context(rt):
  features,options=capture_features(rt,state,op);update('zero_budgets')
  with torch.no_grad():
   deltas=op.latent_deltas(rt,features,options['modality_positions'],state['latent'].to(rt.device),(1,2))
   for k,delta in deltas.items():
    final=replay_delta(rt,state,delta)
    if not torch.equal(final,baseline):raise ValueError('Zero terminal differs at budget'+str(k))
  # Probe only the readout, not an optimizer. Low-rank B remains zero so the
  # proof measures trainability of an initially exact shared recurrent core.
  with torch.no_grad():
   for branch in (op.outer,op.inner):branch.readout.weight.normal_(std=.001)
  torch.cuda.reset_peak_memory_stats();start=time.perf_counter();update('two_terminal_gradient')
  deltas=op.latent_deltas(rt,features,options['modality_positions'],state['latent'].to(rt.device),(1,2))
  low=replay_delta(rt,state,deltas[1],checkpoint_rollout=True)
  high=replay_delta(rt,state,deltas[2],checkpoint_rollout=True)
  loss=(high-target).square().mean()+.5*(low-target).square().mean()+.5*(low-high.detach()).square().mean()
  loss.backward()
  gradients={name:sum(float(p.grad.abs().sum()) for p in branch.low_rank.parameters() if p.grad is not None) for name,branch in (('outer',op.outer),('inner',op.inner))}
  evidence=sum(float(p.grad.abs().sum()) for p in op.encoder.parameters() if p.grad is not None)
  grads=[p.grad for p in op.trainable_parameters() if p.grad is not None]
  if min(gradients.values())<=0 or evidence<=0 or any(not bool(torch.isfinite(g).all()) for g in grads):raise ValueError('Missing/nonfinite nested core/evidence gradient')
  if any(p.grad is not None for p in base):raise ValueError('Frozen base receives gradient')
 if hashes!=op.core_hashes():raise ValueError('functional_call did not restore original shared weights')
 update('passed',zero_terminal_budgets_exact=[1,2],base_parameter_gradients=False,shared_base_cores_exact=True,
  low_rank_gradient_abs_sum=gradients,shared_evidence_gradient_abs_sum=evidence,
  trainable_parameters=sum(p.numel() for p in op.trainable_parameters()),low_rank_parameters=sum(p.numel() for b in (op.outer,op.inner) for p in b.low_rank.parameters()),
  time_s=time.perf_counter()-start,peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3,
  interpretation='Actual2x14 terminal gradient and nested-prefix proof; no optimizer or quality gain')
except Exception as error:update('failed',error=str(error));raise
