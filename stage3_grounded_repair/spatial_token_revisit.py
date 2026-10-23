"""Learnable evidence-conditioned token transport inside frozen shared recurrence.

The coordinate field can relocate content across the full visual grid. It reads
only MMU evidence, current visual features and coordinates, never teacher masks
or pixels. Identity sampling is subtracted explicitly for bit-exact zero init.
This is a capacity experiment, not an established successful repair method.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from token_conditioned_revisit import TokenConditionedRevisit
from evidence_conditioned_revisit import save_operator


class SpatialTokenRevisit(TokenConditionedRevisit):
    def __init__(self,shared_core,hidden_size=2048,level=2,width=128,max_update=.1):
        super().__init__(shared_core,hidden_size,level,width,max_update)
        self.displacement=nn.Linear(width,2)
        nn.init.zeros_(self.displacement.weight);nn.init.zeros_(self.displacement.bias)
        self.operator_kind='spatial_tokenwise'
        self.max_displacement=2.0  # Full normalized grid extent; learned, not scanned.

    def token_field(self,h,positions):
        start,length=positions[0,0].tolist();count=length-1;side=math.isqrt(count)
        if side*side!=count or start+length>h.shape[1] or count<=0:
            raise ValueError('Expected a nonempty square visual token grid')
        if side==1:raise ValueError('Transport needs at least a two by two grid')
        a,b=start+1,start+length
        line=torch.linspace(-1,1,side,device=h.device,dtype=torch.float32)
        yy,xx=torch.meshgrid(line,line,indexing='ij')
        coords=torch.stack((xx,yy),dim=-1).reshape(1,count,2)
        local=self.visual_project(self.norm(h[:1,a:b]).float())
        evidence=self.encoder(self.condition)[:,None,:]
        field=F.silu(local+evidence+self.position_project(coords))
        return field,coords,side,a,b

    def conditioned_hidden(self,h,positions):
        field,coords,side,a,b=self.token_field(h,positions)
        offsets=self.max_displacement*self.displacement(field).tanh()
        source=h[:1,a:b].float().transpose(1,2).reshape(1,h.shape[-1],side,side)
        identity=coords.reshape(1,side,side,2)
        grid=(coords+offsets).reshape(1,side,side,2)
        sampled=F.grid_sample(source,grid,mode='bilinear',padding_mode='border',align_corners=True)
        # Even identity grid_sample need not reproduce h exactly on all devices.
        anchor=F.grid_sample(source,identity,mode='bilinear',padding_mode='border',align_corners=True)
        transport=(sampled-anchor).reshape(1,h.shape[-1],side*side).transpose(1,2)
        gamma,beta=self.film(field).chunk(2,dim=-1)
        gamma,beta=gamma.tanh()*self.max_update,beta.tanh()*self.max_update
        affine=gamma*self.norm(h[:1,a:b]).float()+beta
        delta=torch.zeros_like(h,dtype=torch.float32)
        delta[0,a:b]=(transport+affine)[0]
        return h+delta.to(h.dtype)

    def forward(self,hidden_states,adaln_input,**kwargs):
        if self.mode=='disabled':
            self.executed_core_calls=0;return (hidden_states,)
        if self.mode!='learned':raise ValueError('Spatial capacity operator only implements learned/disabled modes')
        h=hidden_states;positions=kwargs['modality_positions']
        start,length=positions[0,0].tolist()
        mask=torch.zeros(h.shape[:2]+(1,),device=h.device,dtype=h.dtype)
        mask[0,start+1:start+length]=1
        anchor=self.shared_core(hidden_states=h,adaln_input=adaln_input,**kwargs)[0]
        modified=self.conditioned_hidden(h,positions)
        proposal=self.shared_core(hidden_states=modified,adaln_input=adaln_input,**kwargs)[0]
        self.executed_core_calls=2
        return (h+(proposal-anchor)*mask,)


def make_operator(runtime,level=2,width=128,checkpoint=None,core_index=8):
    if not 0<=core_index<len(runtime.original):raise ValueError('Invalid shared core')
    op=SpatialTokenRevisit(runtime.original[core_index],runtime.model.diffusion_head_config.hidden_size,
                           level=level,width=width).to(runtime.device)
    op.shared_core_index=core_index
    if checkpoint:
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False);meta=payload['metadata']
        if payload['level']!=level or payload['width']!=width or meta.get('operator_kind')!='spatial_tokenwise' or meta.get('shared_core_index')!=core_index:
            raise ValueError('Spatial architecture/core differs from checkpoint')
        expected={k for k in op.state_dict() if not k.startswith('shared_core.')}
        if set(payload['operator'])!=expected:raise ValueError('Incomplete spatial checkpoint')
        op.load_state_dict(payload['operator'],strict=False)
    return op
