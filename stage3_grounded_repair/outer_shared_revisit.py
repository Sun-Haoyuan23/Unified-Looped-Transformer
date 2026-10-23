"""MMU-conditioned recurrence in the Qwen backbone, before the flow projection.

Reuses the backbone's final two decoder blocks after the preceding block.
Does not emit extra text tokens. Cache reuse is forbidden for this prototype.
"""
from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.nn import functional as F
from evidence_encoder import EvidenceEncoder


class OuterSharedRevisit(nn.Module):
    def __init__(self,cores,hidden_size=1536,level=2,width=128,max_update=.1):
        super().__init__();self.shared_core=nn.ModuleList(cores)
        self.encoder=EvidenceEncoder(level=level,width=width)
        self.norm=nn.LayerNorm(hidden_size,elementwise_affine=False)
        self.visual_project=nn.Linear(hidden_size,width,bias=False)
        self.position_project=nn.Linear(2,width,bias=False)
        self.film=nn.Linear(width,2*hidden_size)
        nn.init.zeros_(self.film.weight);nn.init.zeros_(self.film.bias)
        self.condition=None;self.positions=None;self.enabled=False;self.executed_core_calls=0
        self.max_update=max_update
        for p in self.shared_core.parameters():p.requires_grad_(False)
    def trainable_parameters(self):
        return [p for n,p in self.named_parameters() if not n.startswith('shared_core.')]
    def run_core(self,h,kwargs):
        if kwargs.get('use_cache',False) or kwargs.get('past_key_value') is not None:
            raise ValueError('Outer recurrence requires cache-free decoder computation')
        for core in self.shared_core:h=core(hidden_states=h,**kwargs)[0]
        return h
    def forward(self,h,**kwargs):
        self.executed_core_calls=0
        if not self.enabled:return h
        start,length=self.positions[0,0].tolist();count=length-1;side=math.isqrt(count)
        if count<=0 or side*side!=count or start+length>h.shape[1]:raise ValueError('Invalid visual token positions')
        a,b=start+1,start+length
        line=torch.linspace(-1,1,side,device=h.device);yy,xx=torch.meshgrid(line,line,indexing='ij')
        coords=torch.stack((xx,yy),dim=-1).reshape(1,count,2)
        field=F.silu(self.visual_project(self.norm(h[:1,a:b]).float())+
                     self.encoder(self.condition)[:,None,:]+self.position_project(coords))
        gamma,beta=self.film(field).chunk(2,-1)
        delta=torch.zeros_like(h)
        delta[0,a:b]=((gamma.tanh()*self.max_update)*self.norm(h[:1,a:b]).float()+beta.tanh()*self.max_update)[0].to(h.dtype)
        anchor=self.run_core(h,kwargs);proposal=self.run_core(h+delta,kwargs)
        mask=torch.zeros(h.shape[:2]+(1,),device=h.device,dtype=h.dtype);mask[0,a:b]=1
        self.executed_core_calls=2*len(self.shared_core)
        return h+(proposal-anchor)*mask


class OuterInsertion(nn.Module):
    def __init__(self,original,repair):super().__init__();self.original=original;self.repair=repair
    def forward(self,hidden_states,*args,**kwargs):
        if args:raise ValueError('Prototype requires keyword decoder context')
        output=self.original(hidden_states=hidden_states,**kwargs)
        return (self.repair(output[0],**kwargs),)+output[1:]


def make_operator(runtime,level=2,width=128):
    layers=runtime.model.showo.model.layers
    if len(layers)<3:raise ValueError('Insufficient outer layers')
    hidden=runtime.model.showo.config.hidden_size
    op=OuterSharedRevisit(list(layers[-2:]),hidden,level,width).to(runtime.device)
    op.outer_insertion_index=len(layers)-3
    return op


@contextmanager
def install(runtime,op):
    layers=runtime.model.showo.model.layers;index=op.outer_insertion_index;original=layers[index]
    cfg=runtime.model.showo.config;old_cache=cfg.use_cache
    try:
        cfg.use_cache=False;layers[index]=OuterInsertion(original,op)
        yield
    finally:
        layers[index]=original;cfg.use_cache=old_cache;op.enabled=False


def replay_outer(runtime,state,op,window=4,checkpoint_rollout=False):
    """Recompute EVERY full velocity; old inner prefix is explicitly invalid."""
    from torch.utils.checkpoint import checkpoint
    from runtime import tensor_hash
    if tensor_hash(state['latent'])!=state['state_fingerprint']:raise ValueError('Changed state')
    grid=state['time_grid'].to(runtime.device)
    if not torch.equal(grid,runtime.grid) or state['CFG']!=runtime.cfg:raise ValueError('Solver/CFG mismatch')
    x=state['latent'].to(runtime.device).clone();options=runtime.inputs(state['prompt'])
    op.positions=options['modality_positions'];initial_condition=op.condition.clone()
    calls=0;extra=0;start=state['flow_index']
    for i in range(start,len(grid)-1):
        active=i<start+window;t=grid[i]
        def velocity(z,now=t,active=active,condition=initial_condition):
            op.enabled=active;op.condition=condition.clone();op.condition[:,3]=now
            return runtime.velocity(z,now,options)
        v=checkpoint(velocity,x,use_reentrant=False) if checkpoint_rollout else velocity(x)
        x=x+(grid[i+1]-grid[i])*v;calls+=1;extra+=2*len(op.shared_core) if active else 0
    return x,dict(network_velocity_calls=calls,outer_backbone_layer_calls=calls*len(runtime.model.showo.model.layers),
        outer_shared_revisit_calls=extra,flow_head_layer_calls=calls*len(runtime.original),
        cached_inner_prefix_used=False,FLOPs=None)
