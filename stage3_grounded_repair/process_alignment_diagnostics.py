"""Read-only direction and connected-reference numerical diagnostics."""
import math
import torch


def direction_stats(predicted, target):
    a=predicted.detach().double().reshape(-1);b=target.detach().double().reshape(-1)
    if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Invalid direction tensors')
    aa=float(a.square().mean());bb=float(b.square().mean());ab=float((a*b).mean())
    result=dict(prediction_rms=math.sqrt(aa),target_rms=math.sqrt(bb),mse=float((a-b).square().mean()))
    result.update(cosine=ab/math.sqrt(aa*bb) if aa>0 and bb>0 else None,
                  relative_target_mse=float((a-b).square().mean())/bb if bb>0 else None)
    return result


def recomputed_reference(branch, hidden, feedback, kwargs, budget):
    """Diagnostic only: recompute the connected reference at each iteration.

    At zero feedback this has the same forward as the cached-reference route.
    A backward difference measures a computation-graph/numerical difference;
    this function is never installed as a training or deployment operator.
    """
    if budget not in (1,2,4) or kwargs.get('use_cache',False) or kwargs.get('past_key_value') is not None:
        raise ValueError('Invalid diagnostic budget or KV cache')
    clean={k:v for k,v in kwargs.items() if k!='hidden_states'}
    def group(h):
        for i in range(len(branch.shared_core)):h=branch.run_core(i,h,clean)
        return h
    work_dtype=torch.float32 if hidden.dtype in (torch.bfloat16,torch.float16) else hidden.dtype
    h=hidden+feedback
    for _ in range(budget):
        reference=group(hidden);response=group(h)
        h=(hidden.to(work_dtype)+(response.to(work_dtype)-reference.to(work_dtype))).to(hidden.dtype)
    return h


def gradient_pair_stats(first, second, names):
    selected=[i for i,n in enumerate(names) if '.low_rank.' in n]
    def values(indices):
        aa=bb=ab=0.
        for i in indices:
            a=first[i];b=second[i]
            if a is not None:aa+=float(a.detach().double().square().sum())
            if b is not None:bb+=float(b.detach().double().square().sum())
            if a is not None and b is not None:ab+=float((a.detach().double()*b.detach().double()).sum())
        return dict(first_l2=math.sqrt(aa),second_l2=math.sqrt(bb),cosine=ab/math.sqrt(aa*bb) if aa>0 and bb>0 else None)
    return dict(all_parameters=values(range(len(names))),shared_low_rank=values(selected),
                public_parameters=values([i for i in range(len(names)) if i not in selected]))
