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
from token_conditioned_revisit import make_operator,save_operator
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
    p.add_argument('--width',type=int,default=128)
    p.add_argument('--window',type=int,default=4)
    p.add_argument('--core-index',type=int,choices=range(10),default=8)
    p.add_argument('--preserve-weight',type=float,default=1.)
    p.add_argument('--rank-weight',type=float,default=.1)
    p.add_argument('--checkpoint-rollout',action=argparse.BooleanOptionalAction,default=True)
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Training resume is not implemented; use a new output directory')
    report=json.loads(args.paired_report.read_text())
    if not report['matched_zero_intervention_passed']:raise RuntimeError('Oracle zero gate failed')
    records=[json.loads(l) for l in args.records.read_text().splitlines() if l.strip()]
    teachers=selected_pairs(report,records)
    if not teachers:raise RuntimeError('NO-GO: no train-split semantic repair preserving both HPS and PickScore')
    correct_states=preservation_states(report,records)
    state_files={f.stem:f for f in args.states.rglob('*.pt')}
    if not (set(teachers)|set(correct_states))<=set(state_files):raise RuntimeError('Training states missing')
    r=Runtime(args.work,vae=False)
    op=make_operator(r,level=args.level,width=args.width,core_index=args.core_index)
    optimizer=torch.optim.AdamW(op.trainable_parameters(),lr=args.lr,weight_decay=0)
    base_core_hash={name:tensor_hash(t) for name,t in r.original[args.core_index].state_dict().items()}
    rng=random.Random(20261004)
    keys=sorted(teachers)
    args.output.mkdir(parents=True,exist_ok=True)
    config=vars(args).copy();config={k:str(v) if isinstance(v,Path) else v for k,v in config.items()}
    config.update(operator_kind='tokenwise',shared_core_index=args.core_index,runtime=r.metadata(),trainable_parameters=sum(p.numel() for p in op.trainable_parameters()),
                  selected_teacher_states=len(teachers),controller='none',
                  paired_report_sha256=file_hash(args.paired_report),records_sha256=file_hash(args.records),
                  training_code_sha256={name:file_hash(Path(__file__).parent/name) for name in
                      ('train_token_repair_operator.py','token_conditioned_revisit.py','runtime.py','evidence_encoder.py',
                       'evidence_conditioned_revisit.py','training_pairs.py')},
                  training_gate=validate_training_pairs(report,records),
                  training_objective='terminal paired-success latent distillation + endpoint ranking + baseline preservation',
                  checkpoint_selection='independent val terminal official evaluation only',
                  successful_teacher_actions={k:v['action'] for k,v in teachers.items()})
    write_json(args.output/'training_config.json',config)
    with (args.output/'train.jsonl').open('a') as stream:
        for step in range(1,args.steps+1):
            preservation_step=step%3==0
            key=rng.choice(correct_states if preservation_step else keys)
            state=torch.load(state_files[key],map_location='cpu',weights_only=False)
            if state['split']!='train' or not state['zero_intervention_test']['passed']:
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
            if step==1:
                with torch.no_grad():
                    zero_final,_=replay(r,state,repair=op,window=args.window)
                if not torch.equal(zero_final,baseline):
                    raise RuntimeError('Training starts from a non-exact zero intervention')
            rng_restore(state['rng'])
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            # Full remaining continuation, with gradients through frozen dynamics.
            # The window restricts where the repairer acts, not where loss is read.
            final,cost=replay(r,state,repair=op,window=args.window,
                               checkpoint_rollout=args.checkpoint_rollout)
            scale=baseline.square().mean().clamp_min(1e-5)
            terminal=F.mse_loss(final,teacher)/scale
            dist_good=F.mse_loss(final,teacher)/scale
            dist_base=F.mse_loss(final,baseline)/scale
            margin=.5*F.mse_loss(teacher,baseline).detach()/scale.detach()
            rank=dist_good.new_zeros(()) if preservation_step else F.relu(margin+dist_good-dist_base)
            preserve=dist_base if preservation_step else dist_base.new_zeros(())
            loss=(args.preserve_weight*preserve if preservation_step else
                  terminal+args.rank_weight*rank)
            loss.backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(op.trainable_parameters(),1.)
            if not torch.isfinite(loss) or not torch.isfinite(grad_norm):raise RuntimeError('Nonfinite repair gradient')
            if step==1 and float(grad_norm)==0:raise RuntimeError('Repair evidence path receives zero gradient')
            optimizer.step()
            torch.cuda.synchronize()
            row={'step':step,'state_id':key,'preservation_step':preservation_step,'loss':float(loss.detach()),
                 'terminal_latent_loss':float(terminal.detach()),'rank_loss':float(rank.detach()),
                 'preserve_loss':float(preserve.detach()),'grad_norm':float(grad_norm),
                 'time_s':time.perf_counter()-start,'peak_gb':torch.cuda.max_memory_allocated()/1024**3}
            stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps(row),flush=True)
            if step%25==0 or step==args.steps:save_operator(op,args.output/f'operator_step{step}.pt',step,config)
    final_core_hash={name:tensor_hash(t) for name,t in r.original[args.core_index].state_dict().items()}
    if final_core_hash!=base_core_hash:raise RuntimeError('Frozen shared core changed')
    write_json(args.output/'training_complete.json',{'steps':args.steps,'first_core_exact':True,
         'validation_required':True,'controller_started':False})


if __name__=='__main__':main()
