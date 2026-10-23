"""Tokenwise MMU-conditioned shared recurrence, with a depth-matched core.

No external instance masks/pixel edits are read at inference. Local visual
content and fixed2D coordinates allow evidence-dependent spatial fields;
the original Transformer still performs the shared global computation.
"""
import math
import torch
from torch import nn
from evidence_conditioned_revisit import ConditionedRevisit,save_operator


class TokenConditionedRevisit(ConditionedRevisit):
    def __init__(self,shared_core,hidden_size=2048,level=2,width=128,max_update=.1):
        super().__init__(shared_core,hidden_size,level,width,max_update)
        self.visual_project=nn.Linear(hidden_size,width,bias=False)
        self.position_project=nn.Linear(2,width,bias=False)
        self.operator_kind='tokenwise'

    def conditioned_affine(self,h,positions):
        start,length=positions[0,0].tolist()
        count=length-1;side=math.isqrt(count)
        if side*side!=count or start+length>h.shape[1]:
            raise ValueError('Expected a square visual token grid')
        coords=h.new_zeros((1,h.shape[1],2),dtype=torch.float32)
        line=torch.linspace(-1,1,side,device=h.device)
        yy,xx=torch.meshgrid(line,line,indexing='ij')
        coords[0,start+1:start+length]=torch.stack((xx,yy),dim=-1).reshape(-1,2)
        local=self.visual_project(self.norm(h[:1]).float())
        evidence=self.encoder(self.condition)[:,None,:]
        field=torch.nn.functional.silu(local+evidence+self.position_project(coords))
        gamma,beta=self.film(field).chunk(2,dim=-1)
        return gamma.tanh()*self.max_update,beta.tanh()*self.max_update


def make_operator(runtime,level=2,width=128,checkpoint=None,core_index=8):
    if not 0<=core_index<len(runtime.original):raise ValueError('Invalid shared core')
    op=TokenConditionedRevisit(runtime.original[core_index],runtime.model.diffusion_head_config.hidden_size,
                              level=level,width=width).to(runtime.device)
    op.shared_core_index=core_index
    if checkpoint:
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        meta=payload['metadata']
        if payload['level']!=level or payload['width']!=width or meta.get('operator_kind')!='tokenwise' or meta.get('shared_core_index')!=core_index:
            raise ValueError('Token architecture/core differs from checkpoint')
        expected={k for k in op.state_dict() if not k.startswith('shared_core.')}
        if set(payload['operator'])!=expected:raise ValueError('Incomplete token checkpoint')
        op.load_state_dict(payload['operator'],strict=False)
    return op
