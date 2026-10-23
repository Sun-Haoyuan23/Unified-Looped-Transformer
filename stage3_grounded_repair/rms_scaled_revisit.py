"""Dimensionless FiLM recurrence: calibrate update units to the current token RMS.

The fixed0.1 bound is relative to each token, not an absolute hidden-unit bound.
Scale is detached. No parameters, extra Transformer calls or external inputs are
introduced; zero heads still recover the complete original path exactly.
"""
import math
import torch
from torch.nn import functional as F
from outer_shared_revisit import OuterSharedRevisit
from token_conditioned_revisit import TokenConditionedRevisit
from two_level_operator import TwoLevelOperator


def hidden_scale(h):
    return h.detach().float().square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)


class RMSOuterRevisit(OuterSharedRevisit):
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
        relative=gamma.tanh()*self.max_update*self.norm(h[:1,a:b]).float()+beta.tanh()*self.max_update
        delta=torch.zeros_like(h)
        delta[0,a:b]=(relative*hidden_scale(h[:1,a:b]))[0].to(h.dtype)
        anchor=self.run_core(h,kwargs);proposal=self.run_core(h+delta,kwargs)
        mask=torch.zeros(h.shape[:2]+(1,),device=h.device,dtype=h.dtype);mask[0,a:b]=1
        self.executed_core_calls=2*len(self.shared_core)
        return h+(proposal-anchor)*mask


class RMSInnerRevisit(TokenConditionedRevisit):
    def conditioned_affine(self,h,positions):
        gamma,beta=super().conditioned_affine(h,positions)
        scale=hidden_scale(h[:1])
        return gamma*scale,beta*scale


def make_operator(runtime,arm,level=2,width=128):
    if arm not in ('outer','inner','joint'):raise ValueError('Unknown recurrence arm')
    layers=runtime.model.showo.model.layers
    outer=RMSOuterRevisit(list(layers[-2:]),runtime.model.showo.config.hidden_size,level,width).to(runtime.device)
    outer.outer_insertion_index=len(layers)-3
    inner=RMSInnerRevisit(runtime.original[8],runtime.model.diffusion_head_config.hidden_size,level,width).to(runtime.device)
    inner.shared_core_index=8
    inner.encoder.load_state_dict(outer.encoder.state_dict())
    inner.position_project.load_state_dict(outer.position_project.state_dict())
    if arm=='joint':inner.encoder=outer.encoder
    return TwoLevelOperator(outer if arm!='inner' else None,inner if arm!='outer' else None)
