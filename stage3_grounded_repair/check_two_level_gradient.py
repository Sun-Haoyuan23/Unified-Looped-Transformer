"""Actual dual-level full-terminal zero/gradient proof; zero optimizer steps."""
from pathlib import Path
import json,time
import torch
from runtime import Runtime,rng_restore,tensor_hash,write_json
from evidence_encoder import condition_tensor
from outer_shared_revisit import make_operator as make_outer
from token_conditioned_revisit import make_operator as make_inner
from two_level_replay import install_levels,replay_levels
from candidate_provenance import file_hash

w=Path('/home/shy_23/iterative_visual_stage1_20261003');r=w/'stage3_grounded_repair'
out=r/'two_level_gradient_v37';out.mkdir()

def update(phase,**kw):write_json(out/'pipeline_status.json',dict(phase=phase,time=time.time(),optimizer_steps=0,**kw))

try:
    update('loading')
    path=next(q for q in (r/'states/train_pilot_minimal_v14').glob('shard*/*_i35.pt') if q.stem=='train_extra_00021_seed101_i35')
    rt=Runtime(w,vae=False);base=list(rt.model.parameters());state=torch.load(path,map_location='cpu',weights_only=False)
    if state['split']!='train':raise ValueError('TRAIN-only interface check')
    torch.manual_seed(20261005);outer=make_outer(rt);inner=make_inner(rt,core_index=8)
    condition=condition_tensor(state['evidence'],state['true_t'],rt.device)
    outer.condition=condition.clone();inner.condition=condition.clone()
    parameters=outer.trainable_parameters()+inner.trainable_parameters()
    hashes={side:{n:tensor_hash(t) for n,t in op.shared_core.state_dict().items()}
            for side,op in (('outer',outer),('inner',inner))}
    write_json(out/'config.json',dict(source_sha256={n:file_hash(r/n) for n in ('two_level_replay.py','check_two_level_gradient.py',
        'outer_shared_revisit.py','token_conditioned_revisit.py','runtime.py')},state_sha256=file_hash(path),
        outer_shared_indices=[26,27],inner_shared_index=8,window=4,old_inner_prefix_used=False,old_KV_used=False))
    with install_levels(rt,outer):
        update('zero_check');rng_restore(state['rng'])
        with torch.no_grad():zero,cost=replay_levels(rt,state,outer,inner)
        baseline=state['baseline_final_latent'].to(rt.device)
        if not torch.equal(zero,baseline):raise ValueError('Joint zero intervention is not strict')
        update('terminal_gradient');rng_restore(state['rng']);started=time.perf_counter()
        final,cost=replay_levels(rt,state,outer,inner,checkpoint_rollout=True)
        target=baseline+.01*torch.randn_like(baseline);(final-target).square().mean().backward()
        gradients={side:sum(float(p.grad.abs().sum()) for p in op.trainable_parameters() if p.grad is not None)
                   for side,op in (('outer',outer),('inner',inner))}
        if any(v<=0 for v in gradients.values()):raise ValueError('A recurrence level has zero gradient')
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):raise ValueError('Nonfinite gradient')
        if any(p.grad is not None for p in base):raise ValueError('Frozen base receives gradients')
    if hashes!={side:{n:tensor_hash(t) for n,t in op.shared_core.state_dict().items()}
                for side,op in (('outer',outer),('inner',inner))}:raise ValueError('Shared core changed')
    update('passed',zero_intervention_latent_exact=True,full_terminal_requires_grad=final.requires_grad,
        adapter_gradient_abs_sum=gradients,base_parameter_gradients=False,shared_cores_exact=True,
        trainable_parameters=sum(p.numel() for p in parameters),cost=cost,
        time_s=time.perf_counter()-started,peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3)
except Exception as error:update('failed',error=str(error));raise
