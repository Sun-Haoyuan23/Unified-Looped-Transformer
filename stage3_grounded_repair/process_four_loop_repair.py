"""Four matched routes with TRAIN-only process loss outside the forward."""
import torch
from torch.nn import functional as F
from anchored_recurrent_repair import branch_inputs
from multibudget_response_repair import shared_response_prefixes
from nonlinear_recurrent_repair import NonlinearRecurrentRepair,normalized_losses
from nonlinear_token_readout import patch_delta

ARMS=('anchor_trainable4','anchor_frozen4','plain_trainable4','no_loop')


def process_tokens(operator,features,positions,arm):
    if arm not in ARMS:raise ValueError('Undeclared process route')
    values=[];hidden={};calls={}
    for name in ('outer','inner'):
        branch=getattr(operator,name);h,kw=features[name]
        field,coords,fb,begin,end=branch_inputs(branch,h,operator.condition,positions)
        if arm=='no_loop':v=h+fb;calls[name]=0
        else:
            prefix,calls[name]=shared_response_prefixes(branch,h,fb,kw,(4,),arm.startswith('anchor'))
            v=prefix[4]
        hidden[name]=v[:1,begin:end]
        values.append(torch.cat((branch.norm(hidden[name]).float(),field,coords),-1))
    return torch.cat(values,-1),hidden,calls


class ProcessFourLoopRepair(NonlinearRecurrentRepair):
    def __init__(self,operator,head,arm):
        if arm not in ARMS:raise ValueError('Undeclared process arm')
        old='no_loop' if arm=='no_loop' else ('frozen_loop' if arm=='anchor_frozen4' else 'trainable_loop')
        super().__init__(operator,head,old);self.process_arm=arm

    def forward_state(self,runtime,state,features,positions):
        if features.get('_provenance')!=(state['state_fingerprint'],state['prompt'],state['flow_index']):
            raise ValueError('Stale original-state features')
        tokens,hidden,self.last_shared_calls=process_tokens(self.operator,features,positions,self.process_arm)
        return patch_delta(runtime,self.head(tokens),state['latent']),hidden

    def latent_delta(self,runtime,state,features,positions):
        return self.forward_state(runtime,state,features,positions)[0]

    def portable_state(self):
        return dict(**super().portable_state(),process_arm=self.process_arm,outer_budget=0 if self.process_arm=='no_loop' else 4,
                    inner_budget=0 if self.process_arm=='no_loop' else 4,reference_connected=self.process_arm.startswith('anchor'))


def normalized_hidden(x):
    return F.layer_norm(x.float(),(x.shape[-1],),eps=1e-5)


def hidden_target(source,target,preservation):
    source=normalized_hidden(source.detach());target=normalized_hidden(target.detach())
    scale=(source.square().mean() if preservation else (source-target).square().mean()).clamp_min(1e-5)
    return target.detach(),scale.detach()


def hidden_loss(predicted,targets,scales):
    return sum((normalized_hidden(predicted[n])-targets[n].detach()).square().mean()/scales[n].detach()
               for n in ('outer','inner'))*.5


def process_loss(delta,target,original,final,baseline,predicted,targets,scales):
    endpoint,physical,terminal=normalized_losses(delta,target['delta'],original,final,target['final'],baseline,target['preservation'])
    hidden=hidden_loss(predicted,targets,scales)
    return endpoint+hidden,physical,terminal,hidden


def process_gradient_groups(repair):
    from nonlinear_recurrent_protocol import gradient_groups
    groups=gradient_groups(repair);groups['feedback']=0.
    for name,p in repair.named_parameters():
        if '.feedback.' in name and p.grad is not None:
            if not torch.isfinite(p.grad).all():raise ValueError('Nonfinite feedback gradient')
            groups['feedback']+=float(p.grad.abs().sum())
    return groups
