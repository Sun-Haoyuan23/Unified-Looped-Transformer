"""B1 only: terminal paired-success supervision, frozen base and MMU.

This uses final latent targets from independently checked, SAME-STATE successful
continuations. It never selects a checkpoint by single-step flow MSE. Official
GenEval is nondifferentiable: measured semantic successes provide teacher targets;
the differentiable terminal loss is explicitly latent outcome distillation.
No controller is trained, and no HPS/Pick gains are fabricated from this loss.
"""
import argparse
import json
import random
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from runtime import Runtime,replay,tensor_hash,write_json,rng_restore
from evidence_encoder import condition_tensor
from state_space_revisit import full_model_context, capture_features
from trainable_recurrent_state import replay_delta
from recurrent_training_protocol import build_operator,nested_loss,STUDIES
from training_pairs import selected_pairs,preservation_states,validate_training_pairs
from candidate_provenance import file_hash


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--states',type=Path,required=True)
    p.add_argument('--records',type=Path,required=True)
    p.add_argument('--paired-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--level',type=int,choices=(0,1,2),default=2)
    p.add_argument('--width',type=int,default=64)
    p.add_argument('--window',type=int,default=1)
    p.add_argument('--preserve-weight',type=float,default=1.)
    p.add_argument('--rank-weight',type=float,default=.1)
    p.add_argument('--checkpoint-rollout',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--study',choices=STUDIES,required=True)
    args=p.parse_args()
    args.arm=args.study.split('_',1)[0]
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Training resume is not implemented; use a new output directory')
    if args.width!=64 or args.window!=1 or args.steps!=100:raise ValueError('Fixed v44 capacity protocol changed')
    report=json.loads(args.paired_report.read_text())
    if not report['matched_zero_intervention_passed']:raise RuntimeError('Oracle zero gate failed')
    records=[json.loads(l) for l in args.records.read_text().splitlines() if l.strip()]
    teachers=selected_pairs(report,records)
    if not teachers:raise RuntimeError('NO-GO: no train-split semantic repair preserving both HPS and PickScore')
    correct_states=preservation_states(report,records)
    state_files={f.stem:f for f in args.states.rglob('*.pt')}
    if not (set(teachers)|set(correct_states))<=set(state_files):raise RuntimeError('Training states missing')
    r=Runtime(args.work,vae=False)
    torch.manual_seed(20261005)
    torch.cuda.manual_seed_all(20261005)
    op=build_operator(r,args.study)
    base_parameters=list(r.model.parameters())
    adapter_init_hashes={k:tensor_hash(v) for k,v in op.state_dict().items() if '.shared_core.' not in k}
    optimizer=torch.optim.AdamW(op.trainable_parameters(),lr=args.lr,weight_decay=0)
    base_core_hash=op.core_hashes()
    rng=random.Random(20261004)
    keys=sorted(teachers)
    args.output.mkdir(parents=True,exist_ok=True)
    config=vars(args).copy();config={k:str(v) if isinstance(v,Path) else v for k,v in config.items()}
    config.update(adapter_initialization_seed=20261005,adapter_initialization_hashes=adapter_init_hashes,operator_kind='recurrent_'+args.study,shared_outer_indices=[26,27] if op.outer is not None else [],shared_inner_indices=[8,9] if op.inner is not None else [],cached_inner_prefix_used=False,runtime=r.metadata(),trainable_parameters=sum(p.numel() for p in op.trainable_parameters()),
                  selected_teacher_states=len(teachers),controller='none',
                  paired_report_sha256=file_hash(args.paired_report),records_sha256=file_hash(args.records),
                  training_code_sha256={name:file_hash(Path(__file__).parent/name) for name in
                      ('train_recurrent_operator.py','trainable_recurrent_state.py','recurrent_training_protocol.py','state_space_revisit.py','runtime.py','evidence_encoder.py','training_pairs.py')},
                  training_gate=validate_training_pairs(report,records),
                  training_objective='teacher-displacement-normalized terminal distillation and ranking; baseline-normalized preservation; same objective in all arms',
                  checkpoint_selection='fixed final step100 before TRAIN fit; no val or controller',
                  successful_teacher_actions={k:v['action'] for k,v in teachers.items()})
    common={k:tensor_hash(v) for k,v in op.encoder.state_dict().items()}
    config['common_evidence_initialization_hashes']=common
    config['shared_evidence_encoder']=args.arm=='joint'
    config['update_units']='rank4 trainable attention in shared recurrent blocks; nested loops1/2 with shared latent readout; one index35 physical update'
    config['maximum_recurrent_loops']=2
    config['intra_loop_self_distillation']=args.study.endswith('_ilsd')
    config['low_rank_rank']=4
    config['pre_feedback_feature_cache']='only unchanged state; every post-update velocity fully recomputed'
    feature_cache={}
    write_json(args.output/'training_config.json',config)
    with full_model_context(r), (args.output/'train.jsonl').open('a') as stream:
        for step in range(1,args.steps+1):
            preservation_step=step%3==0
            key=rng.choice(correct_states if preservation_step else keys)
            state=torch.load(state_files[key],map_location='cpu',weights_only=False)
            if state['split']!='train' or state['flow_index']!=35 or not state['zero_intervention_test']['passed']:
                raise RuntimeError('Invalid training state or split leakage')
            baseline=state['baseline_final_latent'].to(r.device).float()
            if tensor_hash(state['latent'])!=state['state_fingerprint']:
                raise RuntimeError('Captured training state corrupted')
            if preservation_step:
                teacher=baseline
            else:
                teacher_record=teachers[key]
                if teacher_record['state_fingerprint']!=state['state_fingerprint']:
                    raise RuntimeError('Teacher continuation starts from different state')
                if file_hash(teacher_record['final_latent'])!=teacher_record.get('final_latent_sha256'):
                    raise RuntimeError('Teacher terminal file is missing provenance or changed')
                payload=torch.load(teacher_record['final_latent'],map_location=r.device,weights_only=False)
                if payload['state_id']!=key or payload['action']!=teacher_record['action']:
                    raise RuntimeError('Paired terminal teacher identity mismatch')
                teacher=payload['latent'].float()
            op.condition=condition_tensor(state['evidence'],state['true_t'],r.device)
            cache_hit=key in feature_cache
            if not cache_hit:
                feature_cache[key],_=capture_features(r,state,op)
            if step==1:
                with torch.no_grad():
                    zero_delta=op.latent_deltas(r,feature_cache[key],r.inputs(state['prompt'])['modality_positions'],state['latent'].to(r.device),(2,))[2]
                    zero_final=replay_delta(r,state,zero_delta)
                if not torch.equal(zero_final,baseline):
                    raise RuntimeError('Training starts from a non-exact zero intervention')
            rng_restore(state['rng'])
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            # Full remaining continuation, with gradients through frozen dynamics.
            # The window restricts where the repairer acts, not where loss is read.
            budgets=(1,2) if args.study.endswith('_ilsd') else (2,)
            deltas=op.latent_deltas(r,feature_cache[key],r.inputs(state['prompt'])['modality_positions'],state['latent'].to(r.device),budgets)
            final=replay_delta(r,state,deltas[2],checkpoint_rollout=args.checkpoint_rollout)
            student=replay_delta(r,state,deltas[1],checkpoint_rollout=args.checkpoint_rollout) if 1 in deltas else None
            loss,parts=nested_loss(final,student,teacher,baseline,preservation_step,step,args.steps,args.rank_weight)
            terminal,rank,preserve=parts['terminal'],parts['rank'],parts['preserve']
            loss.backward()
            if any(p.grad is not None for p in base_parameters):raise RuntimeError('Frozen base receives gradients')
            grad_norm=torch.nn.utils.clip_grad_norm_(op.trainable_parameters(),1.)
            if not torch.isfinite(loss) or not torch.isfinite(grad_norm):raise RuntimeError('Nonfinite repair gradient')
            if step==1 and float(grad_norm)==0:raise RuntimeError('Repair evidence path receives zero gradient')
            optimizer.step()
            torch.cuda.synchronize()
            row={'step':step,'state_id':key,'preservation_step':preservation_step,'loss':float(loss.detach()),
                 'terminal_latent_loss':float(terminal.detach()),'rank_loss':float(rank.detach()),
                 'preserve_loss':float(preserve.detach()),'grad_norm':float(grad_norm),
                 'pre_feedback_features_cached':cache_hit,'post_update_prefix_or_KV_used':False,
                 'student_loss':float(parts['student'].detach()),'distillation_loss':float(parts['distill'].detach()),
                 'student_ground_truth_fraction':parts['ground_truth_fraction'],
                 'low_rank_gradient_abs_sum':sum(float(p.grad.abs().sum()) for b in (op.outer,op.inner) if b is not None for p in b.low_rank.parameters() if p.grad is not None),
                 'time_s':time.perf_counter()-start,'peak_gb':torch.cuda.max_memory_allocated()/1024**3}
            stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps(row),flush=True)
            if step%25==0 or step==args.steps:
                payload=dict(operator={k:v.detach().cpu() for k,v in op.state_dict().items()
                                       if '.shared_core.' not in k},
                             metadata=config,step=step,level=args.level,width=args.width)
                torch.save(payload,args.output/f'operator_step{step}.pt')
    final_core_hash=op.core_hashes()
    if final_core_hash!=base_core_hash:raise RuntimeError('Frozen shared core changed')
    write_json(args.output/'training_complete.json',{'steps':args.steps,'outer_core_exact':True,'base_parameter_gradients':False,
         'validation_required':True,'controller_started':False})


if __name__=='__main__':main()
