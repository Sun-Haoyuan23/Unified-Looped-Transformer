"""TRAIN-only supervision of existing nested recurrent states; no new parameters.

Hooks observe the exact existing forward. They are removed before the physical
state is advanced, so original full flow evaluations remain untouched.
"""
from contextlib import contextmanager
import math
import torch
from torch.nn import functional as F


@contextmanager
def recurrent_trace(operator, budgets=(1,2)):
    if tuple(budgets)!=(1,2):
        raise ValueError('Fixed process protocol requires both exact nested prefixes')
    trace, handles = {}, []
    try:
        for name,branch in (('outer',operator.outer),('inner',operator.inner)):
            if branch is None:
                continue
            trace[name]=[]
            def save(module, args, output, key=name):
                trace[key].append(output[0])
                if len(trace[key])>2:
                    raise ValueError('Unexpected extra recurrent visit inside trace')
            handles.append(branch.shared_core[-1].register_forward_hook(save))
        if not trace:
            raise ValueError('No shared recurrent branch')
        yield trace
        if any(len(v)!=2 for v in trace.values()):
            raise ValueError('Missing recurrent prefix in trace')
    finally:
        for handle in handles:
            handle.remove()


def visual_span(hidden, positions):
    if positions.shape!=(2,1,2) or hidden.ndim!=3 or hidden.shape[0]!=2:
        raise ValueError('Expected B2 CFG hidden states and T2I modality positions')
    offset,length=positions[0,0].tolist()
    begin,end=offset+1,offset+length
    count=length-1
    if count<=0 or math.isqrt(count)**2!=count or begin<0 or end>hidden.shape[1]:
        raise ValueError('Invalid visual token span')
    return hidden[:1,begin:end]


def normalized_visual(hidden, positions):
    visual=visual_span(hidden,positions).float()
    return F.layer_norm(visual,(visual.shape[-1],))


def frozen_feature_normalizers(initial_trace, target_features, positions):
    """One immutable scale per state/branch/prefix from the untrained operator."""
    scales={}
    for branch,states in initial_trace.items():
        target=normalized_visual(target_features[branch].detach(),positions)
        if len(states)!=2:
            raise ValueError('Require exactly two initial prefixes')
        scales[branch]=[
            F.mse_loss(normalized_visual(state.detach(),positions),target).clamp_min(1e-5).detach()
            for state in states]
    return scales


def process_terms(deltas, trace, original, target_input, target_features, positions, preservation, scales):
    if set(deltas)!={1,2} or target_input.shape!=original.shape or original.ndim!=4 or original.shape[0]!=2:
        raise ValueError('State target and nested budgets mismatch')
    if set(trace)!=set(scales) or any(len(v)!=2 for v in trace.values()):
        raise ValueError('Feature trace/scales mismatch')
    target=target_input.detach()
    denominator=(original.square().mean() if preservation else (target-original).square().mean()).detach().clamp_min(1e-5)
    state_losses=[]
    for budget in (1,2):
        delta=deltas[budget]
        if delta.shape!=original.shape or not torch.equal(delta[0],delta[1]):
            raise ValueError('Physical update must be identical in both CFG branches')
        state_losses.append(F.mse_loss(original+delta,target)/denominator)
    feature_losses=[]
    for branch,states in trace.items():
        target_hidden=normalized_visual(target_features[branch].detach(),positions)
        for index,hidden in enumerate(states):
            scale=scales[branch][index]
            if scale.ndim or scale.requires_grad or not bool(torch.isfinite(scale)) or float(scale)<=0:
                raise ValueError('Feature scales must be positive, finite and frozen')
            feature_losses.append(F.mse_loss(normalized_visual(hidden,positions),target_hidden)/scale)
    return dict(state=torch.stack(state_losses).mean(),hidden=torch.stack(feature_losses).mean())
