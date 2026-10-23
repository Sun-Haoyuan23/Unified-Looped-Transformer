"""Real-model terminal gradient dry-run; no optimizer or parameter update."""
import argparse
import time
from pathlib import Path
import torch
from runtime import Runtime,replay,tensor_hash,rng_restore,write_json
from evidence_encoder import condition_tensor
from token_conditioned_revisit import make_operator


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--core-index',type=int,default=8)
    args=parser.parse_args()
    r=Runtime(args.work,vae=False)
    base_parameters=list(r.model.parameters())
    state=torch.load(args.state,map_location='cpu',weights_only=False)
    op=make_operator(r,level=2,core_index=args.core_index)
    op.condition=condition_tensor(state['evidence'],state['true_t'],r.device)
    frozen_hash={k:tensor_hash(v) for k,v in r.original[args.core_index].state_dict().items()}
    baseline=state['baseline_final_latent'].to(r.device)
    rng_restore(state['rng'])
    with torch.no_grad():
        zero,_=replay(r,state,repair=op)
    if not torch.equal(zero,baseline):raise RuntimeError('Unwrapped generation changes zero inference')
    rng_restore(state['rng'])
    started=time.perf_counter()
    final,cost=replay(r,state,repair=op,checkpoint_rollout=True)
    if not final.requires_grad:raise RuntimeError('Actual full terminal rollout is detached')
    target=baseline+.01*torch.randn_like(baseline)
    loss=(final-target).square().mean()
    loss.backward()
    gradients=[p.grad for p in op.trainable_parameters() if p.grad is not None]
    total=sum(float(g.abs().sum()) for g in gradients)
    if total<=0 or any(not bool(torch.isfinite(g).all()) for g in gradients):
        raise RuntimeError('Actual evidence adapter has no finite nonzero terminal gradient')
    if any(p.grad is not None for p in base_parameters):
        raise RuntimeError('Frozen base model receives parameter gradient')
    if {k:tensor_hash(v) for k,v in r.original[args.core_index].state_dict().items()}!=frozen_hash:
        raise RuntimeError('Dry-run changed frozen core')
    write_json(args.output,{'status':'passed','state_id':state['state_id'],
        'terminal_requires_grad':True,'adapter_gradient_abs_sum':total,
        'zero_intervention_latent_exact':True,'base_parameter_gradients':False,
        'optimizer_steps':0,'operator_kind':'tokenwise','shared_core_index':args.core_index,
        'trainable_parameters':sum(p.numel() for p in op.trainable_parameters()),'frozen_core_exact':True,'cost':cost,
        'time_s':time.perf_counter()-started,
        'peak_gb':torch.cuda.max_memory_allocated()/1024**3})
    print(args.output.read_text(),flush=True)
