"""Shared Transformer response recurrence anchored to the original token state.

For one shared group G and original h0, update h <- h0 + G(h)-G(h0).
The reference remains connected to trainable shared weights. It is evaluated
once per invocation and reused across nested budgets, never across states.
No target tensors or state identities enter this forward.
"""
import math
import torch
from torch.nn import functional as F
from nonlinear_recurrent_repair import NonlinearRecurrentRepair
from nonlinear_token_readout import patch_delta


def revisit_hidden(branch,hidden,feedback,kwargs,budgets=(1,2),anchored=True):
    if not budgets or len(set(budgets))!=len(budgets) or min(budgets)<1 or max(budgets)>2:
        raise ValueError('Declared nested budgets are one and two')
    if kwargs.get('use_cache',False) or kwargs.get('past_key_value') is not None:
        raise ValueError('No recurrent KV cache')
    clean={k:v for k,v in kwargs.items() if k!='hidden_states'};calls=0
    def group(h):
        nonlocal calls
        for i in range(len(branch.shared_core)):
            h=branch.run_core(i,h,clean);calls+=1
        return h
    # This is a same-forward reference, not a stale state or detached teacher.
    reference=group(hidden) if anchored else None
    h=hidden+feedback;result={}
    for budget in range(1,max(budgets)+1):
        h=group(h)
        if anchored:
            work_dtype=torch.float32 if hidden.dtype in (torch.float16,torch.bfloat16) else hidden.dtype
            response=h.to(work_dtype)-reference.to(work_dtype)
            h=(hidden.to(work_dtype)+response).to(hidden.dtype)
        if budget in budgets:result[budget]=h
    return result,calls


def branch_inputs(branch,hidden,condition,positions):
    offset,length=positions[0,0].tolist();begin,end=offset+1,offset+length;count=length-1;side=math.isqrt(count)
    if hidden.shape[0]!=2 or count<=0 or side*side!=count or end>hidden.shape[1]:
        raise ValueError('Invalid conditional visual token span')
    line=torch.linspace(-1,1,side,device=hidden.device);yy,xx=torch.meshgrid(line,line,indexing='ij')
    coords=torch.stack((xx,yy),-1).reshape(1,count,2)
    field=F.silu(branch.visual_project(branch.norm(hidden[:1,begin:end]).float())+
                 branch.encoder(condition)[:,None]+branch.position_project(coords))
    feedback=torch.zeros_like(hidden);feedback[0,begin:end]=branch.feedback(field)[0].to(hidden.dtype)
    return field,coords,feedback,begin,end


def anchored_token_features(operator,features,positions,budgets=(1,2)):
    values={k:[] for k in budgets};calls={}
    for name in ('outer','inner'):
        branch=getattr(operator,name);hidden,kwargs=features[name]
        field,coords,feedback,begin,end=branch_inputs(branch,hidden,operator.condition,positions)
        hs,calls[name]=revisit_hidden(branch,hidden,feedback,kwargs,budgets,True)
        for k,h in hs.items():values[k].append(torch.cat((branch.norm(h[:1,begin:end]).float(),field,coords),-1))
    return {k:torch.cat(v,-1) for k,v in values.items()},calls


class AnchoredRecurrentRepair(NonlinearRecurrentRepair):
    def __init__(self,operator,head,arm,loops=2):
        if arm=='no_loop' or loops not in (1,2):raise ValueError('Require declared anchored loop route')
        super().__init__(operator,head,arm);self.loops=loops

    def latent_delta(self,runtime,state,features,positions):
        if features.get('_provenance')!=(state['state_fingerprint'],state['prompt'],state['flow_index']):
            raise ValueError('Features belong to another physical state')
        tokens,calls=anchored_token_features(self.operator,features,positions,(self.loops,))
        self.last_shared_calls=calls
        return patch_delta(runtime,self.head(tokens[self.loops]),state['latent'])

    def portable_state(self):
        return dict(**super().portable_state(),route='anchored_response',loops=self.loops)
