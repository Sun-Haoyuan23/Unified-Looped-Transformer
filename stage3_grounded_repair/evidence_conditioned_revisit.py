"""Frozen shared visual core with zero-exact, evidence-conditioned repair.

Zero FiLM alone does NOT make a new Transformer application an identity.
We use h + F(h + gamma*LN(h) + beta) - F(h). Both F calls reuse the
original block weights. All added parameters live only in the evidence path.
The anchor is recomputed on the current h; it is not a stale trajectory cache.
"""
import torch
from torch import nn
from evidence_encoder import EvidenceEncoder


class ConditionedRevisit(nn.Module):
    def __init__(self, shared_core, hidden_size=2048, level=1, width=128, max_update=.1):
        super().__init__()
        self.shared_core=shared_core
        self.encoder=EvidenceEncoder(level=level,width=width)
        self.film=nn.Linear(width,2*hidden_size)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        self.norm=nn.LayerNorm(hidden_size,elementwise_affine=False)
        self.max_update=max_update
        self.condition=None
        self.mode='learned'
        self.generic_gain=.05
        self.probe_beta=None
        self.executed_core_calls=2
        for p in self.shared_core.parameters(): p.requires_grad_(False)

    def set_time(self, value):
        self.condition=self.condition.clone()
        self.condition[:,3]=value

    def trainable_parameters(self):
        return [p for name,p in self.named_parameters() if not name.startswith('shared_core.')]

    def conditioned_affine(self, h, positions):
        affine=self.film(self.encoder(self.condition))
        gamma,beta=affine.chunk(2,dim=-1)
        return gamma[:,None,:].tanh()*self.max_update,beta[:,None,:].tanh()*self.max_update

    def forward(self, hidden_states, adaln_input, **kwargs):
        h=hidden_states
        if self.mode=='disabled':
            self.executed_core_calls=0
            return (h,)
        positions=kwargs['modality_positions']
        mask=torch.zeros(h.shape[:2]+(1,),device=h.device,dtype=h.dtype)
        # Conditional CFG branch only. Text and time token states are untouched.
        start,length=positions[0,0].tolist()
        mask[0,start+1:start+length]=1
        anchor=self.shared_core(hidden_states=h,adaln_input=adaln_input,**kwargs)[0]
        if self.mode=='generic':
            self.executed_core_calls=1
            return (h+self.generic_gain*(anchor-h)*mask,)
        if self.mode=='probe':
            gamma=torch.zeros_like(h[:1,:1])
            value=self.probe_beta.to(h.device,h.dtype)
            if value.ndim==1:beta=value[None,None,:]*self.generic_gain
            elif value.ndim==2 and value.shape==h.shape[1:]:beta=value[None,:,:]*self.generic_gain
            else:raise ValueError('Probe must be a semantic vector or a token-aligned semantic field')
        else:
            gamma,beta=self.conditioned_affine(h,positions)
        modified=h+(gamma.to(h.dtype)*self.norm(h)+beta.to(h.dtype))*mask
        proposal=self.shared_core(hidden_states=modified,adaln_input=adaln_input,**kwargs)[0]
        self.executed_core_calls=2
        return (h+(proposal-anchor)*mask,)


def make_operator(runtime, level=1, width=128, checkpoint=None):
    operator=ConditionedRevisit(runtime.original[2],runtime.model.diffusion_head_config.hidden_size,
                                level=level,width=width).to(runtime.device)
    if checkpoint:
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if payload['level']!=level or payload['width']!=width:
            raise ValueError('Evidence architecture and checkpoint differ')
        expected={k for k in operator.state_dict() if not k.startswith('shared_core.')}
        if set(payload['operator'])!=expected:
            raise ValueError('Repair checkpoint has missing or extra nonshared weights')
        operator.load_state_dict(payload['operator'],strict=False)
    return operator


def save_operator(operator,path,step,metadata):
    # Shared weights are identified by base hash, not duplicated in checkpoints.
    state={k:v.detach().cpu() for k,v in operator.state_dict().items()
           if not k.startswith('shared_core.')}
    torch.save({'operator':state,'step':step,'level':operator.encoder.level,
                'width':operator.encoder.output_size,'metadata':metadata},path)
