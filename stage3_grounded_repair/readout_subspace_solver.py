"""FP64 fixed-feature, group-balanced least-squares diagnostic only."""
import math
import torch

def patch_target_from_permutation(physical, permutation, shape):
    flat=physical.reshape(-1);p=permutation.reshape(-1).long()
    if p.numel()!=flat.numel() or not torch.equal(torch.sort(p).values,torch.arange(p.numel(),device=p.device)):
        raise ValueError('Unpatchify must be a bijection')
    out=torch.empty_like(flat);out[p]=flat
    return out.reshape(shape)

def weighted_block(activation,target,scale,state_weight):
    if activation.ndim!=2 or target.ndim!=2 or activation.shape[0]!=target.shape[0]:raise ValueError('Bad patch matrix')
    if scale<=0 or state_weight<=0:raise ValueError('Positive scale/weight required')
    x=torch.cat((activation.double().cpu(),torch.ones((activation.shape[0],1),dtype=torch.float64)),1)
    y=target.double().cpu();weight=math.sqrt(state_weight/(y.numel()*scale))
    return x*weight,y*weight

def solve_fixed_subspace(x,y):
    if x.device.type!='cpu' or y.device.type!='cpu' or x.dtype!=torch.float64 or y.dtype!=torch.float64:
        raise ValueError('CPU FP64 matrices required')
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('Nonfinite matrices')
    rcond=max(x.shape)*torch.finfo(torch.float64).eps
    u,s,vh=torch.linalg.svd(x,full_matrices=False);cutoff=float(s[0])*rcond;keep=s>cutoff
    c=vh[keep].T@((u[:,keep].T@y)/s[keep,None])
    residual=x@c-y;normal=x.T@residual
    ratio=float(normal.norm()/(x.norm()*residual.norm()).clamp_min(torch.finfo(torch.float64).tiny))
    return c,dict(rcond=rcond,cutoff=cutoff,numerical_rank=int(keep.sum()),singular_values=s.tolist(),
        weighted_objective=float(residual.square().sum()),normal_equation_abs_l2=float(normal.norm()),
        normal_equation_relative_l2=ratio,coefficient_l2=float(c.norm()),rows=x.shape[0],columns=x.shape[1],outputs=y.shape[1])
