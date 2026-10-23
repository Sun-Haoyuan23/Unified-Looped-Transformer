"""Outer-backbone actual full-terminal gradient check; no optimizer."""
import pathlib,json,time
import torch
from runtime import Runtime,tensor_hash,rng_restore,write_json
from evidence_encoder import condition_tensor
from outer_shared_revisit import make_operator,install,replay_outer
from candidate_provenance import file_hash

w=pathlib.Path('/home/shy_23/iterative_visual_stage1_20261003');r=w/'stage3_grounded_repair'
out=r/'outer_revisit_gradient_v31';out.mkdir()
status=out/'pipeline_status.json'
def update(phase,**kw):write_json(status,dict(phase=phase,time=time.time(),optimizer_steps=0,**kw))
try:
    update('loading')
    state_path=next(p for p in (r/'states/train_pilot_minimal_v14').rglob('*_i35.pt') if p.stem=='train_extra_00021_seed101_i35')
    rt=Runtime(w,vae=False);base=list(rt.model.parameters());state=torch.load(state_path,map_location='cpu',weights_only=False)
    if state['split']!='train':raise ValueError('TRAIN-only interface check')
    torch.manual_seed(20261005);torch.cuda.manual_seed_all(20261005)
    op=make_operator(rt);op.condition=condition_tensor(state['evidence'],state['true_t'],rt.device)
    hashes={name:tensor_hash(t) for name,t in op.shared_core.state_dict().items()}
    write_json(out/'config.json',dict(state_sha256=file_hash(state_path),
        code_sha256={n:file_hash(r/n) for n in ('outer_shared_revisit.py','check_outer_rollout_gradient.py','runtime.py')},
        outer_layers=len(rt.model.showo.model.layers),outer_core_indices=list(range(len(rt.model.showo.model.layers)-2,len(rt.model.showo.model.layers))),
        flow_layers=len(rt.original),semantic_encoder_layers=len(rt.model.und_trans.layers),
        cached_inner_prefix_used=False,operator_trainable_parameters=sum(p.numel() for p in op.trainable_parameters())))
    with install(rt,op):
        rng_restore(state['rng']);update('zero_check')
        with torch.no_grad():zero,cost=replay_outer(rt,state,op)
        baseline=state['baseline_final_latent'].to(rt.device)
        if not torch.equal(zero,baseline):raise RuntimeError('Full outer zero replay differs from captured baseline')
        rng_restore(state['rng']);update('terminal_gradient');started=time.perf_counter()
        final,cost=replay_outer(rt,state,op,checkpoint_rollout=True)
        target=baseline+.01*torch.randn_like(baseline)
        (final-target).square().mean().backward()
        grads=[p.grad for p in op.trainable_parameters() if p.grad is not None]
        total=sum(float(g.abs().sum()) for g in grads)
        if total<=0 or any(not bool(torch.isfinite(g).all()) for g in grads):raise ValueError('No finite outer gradient')
        if any(p.grad is not None for p in base):raise ValueError('Frozen base has parameter gradients')
    if hashes!={n:tensor_hash(t) for n,t in op.shared_core.state_dict().items()}:raise ValueError('Shared outer core changed')
    update('passed',zero_intervention_latent_exact=True,full_terminal_requires_grad=final.requires_grad,
        adapter_gradient_abs_sum=total,base_parameter_gradients=False,frozen_core_exact=True,
        trainable_parameters=sum(p.numel() for p in op.trainable_parameters()),cost=cost,
        time_s=time.perf_counter()-started,peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3)
except Exception as error:update('failed',error=str(error));raise
