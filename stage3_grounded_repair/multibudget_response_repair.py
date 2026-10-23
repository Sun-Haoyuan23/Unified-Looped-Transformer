"""User-authorized independent outer/inner shared budgets one, two and four.

This extends v75 without changing its frozen implementation or results.
Nested prefixes share the exact same parameters, feedback and connected anchor.
"""
import torch
from anchored_recurrent_repair import branch_inputs
from nonlinear_recurrent_repair import NonlinearRecurrentRepair
from nonlinear_token_readout import patch_delta


def shared_response_prefixes(branch,hidden,feedback,kwargs,budgets=(1,2,4),anchored=True):
    if not budgets or len(set(budgets))!=len(budgets) or any(k not in (1,2,4) for k in budgets):
        raise ValueError('Predeclared budgets are one, two and four')
    if kwargs.get('use_cache',False) or kwargs.get('past_key_value') is not None:raise ValueError('No recurrent KV cache')
    clean={k:v for k,v in kwargs.items() if k!='hidden_states'};calls=0
    def group(h):
        nonlocal calls
        for i in range(len(branch.shared_core)):h=branch.run_core(i,h,clean);calls+=1
        return h
    reference=group(hidden) if anchored else None;h=hidden+feedback;outputs={}
    work_dtype=torch.float32 if hidden.dtype in (torch.float16,torch.bfloat16) else hidden.dtype
    for k in range(1,max(budgets)+1):
        h=group(h)
        if anchored:h=(hidden.to(work_dtype)+(h.to(work_dtype)-reference.to(work_dtype))).to(hidden.dtype)
        if k in budgets:outputs[k]=h
    return outputs,calls


def budget_token_features(operator,features,positions,budget_pairs=((1,1),(2,2),(4,4)),anchored=True):
    if not budget_pairs or len(set(budget_pairs))!=len(budget_pairs):raise ValueError('Unique declared outer/inner pairs required')
    values={};calls={}
    for column,name in enumerate(('outer','inner')):
        branch=getattr(operator,name);h,kw=features[name]
        field,coords,feedback,begin,end=branch_inputs(branch,h,operator.condition,positions)
        budgets=tuple(sorted({p[column] for p in budget_pairs}))
        hs,calls[name]=shared_response_prefixes(branch,h,feedback,kw,budgets,anchored)
        values[name]={k:torch.cat((branch.norm(v[:1,begin:end]).float(),field,coords),-1) for k,v in hs.items()}
    return {p:torch.cat((values['outer'][p[0]],values['inner'][p[1]]),-1) for p in budget_pairs},calls


class MultiBudgetResponseRepair(NonlinearRecurrentRepair):
    def __init__(self,operator,head,arm,outer_budget=4,inner_budget=4,anchored=True):
        if arm=='no_loop':raise ValueError('Use the original no-loop matched control')
        super().__init__(operator,head,arm)
        if outer_budget not in (1,2,4) or inner_budget not in (1,2,4):raise ValueError('Undeclared loop budgets')
        self.budgets=(outer_budget,inner_budget);self.anchored=anchored

    def latent_deltas(self,runtime,state,features,positions,budget_pairs):
        if features.get('_provenance')!=(state['state_fingerprint'],state['prompt'],state['flow_index']):raise ValueError('Stale physical features')
        tokens,self.last_shared_calls=budget_token_features(self.operator,features,positions,budget_pairs,self.anchored)
        return {p:patch_delta(runtime,self.head(v),state['latent']) for p,v in tokens.items()}

    def latent_delta(self,runtime,state,features,positions):
        return self.latent_deltas(runtime,state,features,positions,(self.budgets,))[self.budgets]

    def portable_state(self):
        return dict(**super().portable_state(),route='anchored_response' if self.anchored else 'plain_shared',outer_budget=self.budgets[0],inner_budget=self.budgets[1])
