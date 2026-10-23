"""Actual tied-memory joint gradient, with explicit nonzero-head probe and zero check."""
from pathlib import Path
import time
import torch
from runtime import Runtime,rng_restore,tensor_hash,write_json
from evidence_encoder import condition_tensor
from rms_scaled_revisit import make_operator
from two_level_replay import install_levels,replay_levels
from candidate_provenance import file_hash

w=Path('/home/shy_23/iterative_visual_stage1_20261003');r=w/'stage3_grounded_repair';out=r/'rms_shared_gradient_v41';out.mkdir()
def update(phase,**kw):write_json(out/'pipeline_status.json',dict(phase=phase,time=time.time(),optimizer_steps=0,**kw))
try:
    update('loading');rt=Runtime(w,vae=False);base=list(rt.model.parameters())
    path=next(q for q in (r/'states/train_pilot_minimal_v14').glob('shard*/*_i35.pt') if q.stem=='train_extra_00021_seed101_i35')
    state=torch.load(path,map_location='cpu',weights_only=False);torch.manual_seed(20261005)
    op=make_operator(rt,'joint');op.condition=condition_tensor(state['evidence'],state['true_t'],rt.device)
    hashes={n:tensor_hash(t) for n,t in op.shared_core.state_dict().items()}
    write_json(out/'config.json',dict(source_sha256={n:file_hash(r/n) for n in ('check_rms_shared_gradient.py','rms_scaled_revisit.py','two_level_operator.py','two_level_replay.py')},
        state_sha256=file_hash(path),nonzero_film_probe_std=.001,training_updates=0))
    with install_levels(rt,op.outer):
        rng_restore(state['rng']);update('zero_check')
        with torch.no_grad():zero,cost=replay_levels(rt,state,op.outer,op.inner)
        baseline=state['baseline_final_latent'].to(rt.device)
        if not torch.equal(zero,baseline):raise ValueError('Shared-memory zero differs')
        with torch.no_grad():
            for branch in (op.outer,op.inner):branch.film.weight.normal_(std=.001)
        rng_restore(state['rng']);update('terminal_gradient');started=time.perf_counter()
        final,cost=replay_levels(rt,state,op.outer,op.inner,checkpoint_rollout=True)
        target=baseline+.01*torch.randn_like(baseline);(final-target).square().mean().backward()
        grads=[p.grad for p in op.trainable_parameters() if p.grad is not None]
        memory=sum(float(p.grad.abs().sum()) for p in op.encoder.parameters() if p.grad is not None)
        if memory<=0 or any(not bool(torch.isfinite(g).all()) for g in grads):raise ValueError('Shared memory gradient missing/nonfinite')
        if any(p.grad is not None for p in base):raise ValueError('Frozen base gradient')
    if hashes!={n:tensor_hash(t) for n,t in op.shared_core.state_dict().items()}:raise ValueError('Core changed')
    update('passed',zero_intervention_latent_exact=True,shared_memory_gradient_abs_sum=memory,
        adapter_gradient_abs_sum=sum(float(g.abs().sum()) for g in grads),base_parameter_gradients=False,
        shared_cores_exact=True,nonzero_head_probe=True,trainable_parameters=sum(p.numel() for p in op.trainable_parameters()),
        time_s=time.perf_counter()-started,peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3,cost=cost)
except Exception as error:update('failed',error=str(error));raise
