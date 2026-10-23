"""Full-velocity replay with outer/inner shared recurrence and bound checkpoint state."""
from contextlib import contextmanager,nullcontext
import torch
from torch import nn
from outer_shared_revisit import install


@contextmanager
def install_levels(runtime,outer=None):
    config=runtime.model.showo.config;previous=config.use_cache
    try:
        config.use_cache=False
        with install(runtime,outer) if outer is not None else nullcontext():yield
    finally:
        config.use_cache=previous
        runtime.model.diffusion_head_a=nn.ModuleList(runtime.original)
        if outer is not None:outer.enabled=False


def replay_levels(runtime,state,outer=None,inner=None,window=4,checkpoint_rollout=False):
    from torch.utils.checkpoint import checkpoint
    from runtime import tensor_hash
    if tensor_hash(state['latent'])!=state['state_fingerprint']:raise ValueError('Changed captured state')
    grid=state['time_grid'].to(runtime.device)
    if not torch.equal(grid,runtime.grid) or state['CFG']!=runtime.cfg:raise ValueError('Changed solver/CFG')
    options=runtime.inputs(state['prompt']);x=state['latent'].to(runtime.device).clone()
    if outer is not None:outer.positions=options['modality_positions']
    outer_condition=outer.condition.clone() if outer is not None else None
    inner_condition=inner.condition.clone() if inner is not None else None
    start=state['flow_index'];calls=0;outer_calls=0;inner_calls=0
    for i in range(start,len(grid)-1):
        active=i<start+window;t=grid[i]
        layers=runtime.original[:8]+[inner]+runtime.original[8:] if active and inner is not None else runtime.original
        def velocity(z,now=t,active=active,layers=tuple(layers),oc=outer_condition,ic=inner_condition):
            runtime.model.diffusion_head_a=nn.ModuleList(layers)
            if outer is not None:
                outer.enabled=active;outer.condition=oc.clone();outer.condition[:,3]=now
            if inner is not None and active:
                inner.mode='learned';inner.condition=ic.clone();inner.set_time(float(now))
            return runtime.velocity(z,now,options)
        v=checkpoint(velocity,x,use_reentrant=False) if checkpoint_rollout else velocity(x)
        x=x+(grid[i+1]-grid[i])*v;calls+=1
        if active:
            outer_calls+=2*len(outer.shared_core) if outer is not None else 0
            inner_calls+=2 if inner is not None else 0
    return x,dict(network_velocity_calls=calls,
        outer_backbone_layer_calls=calls*len(runtime.model.showo.model.layers),outer_shared_revisit_calls=outer_calls,
        flow_head_layer_calls=calls*len(runtime.original),inner_shared_revisit_calls=inner_calls,
        semantic_encoder_layer_calls_nominal=calls*len(runtime.model.und_trans.layers),FLOPs=None)
