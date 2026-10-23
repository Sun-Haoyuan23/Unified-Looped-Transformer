"""Causal offline reflection with coherent clean/noise orientation.

Previous bridge reflected clean but kept the original residual orientation.
Here reflection is deterministic from the SAME captured state/original seed;
no seed or terminal-outcome search is used. Full first velocity is recomputed.
"""
import torch
from torch import nn
from fm_rebridge import linear_rebridge
from runtime import tensor_hash


def reflected_input(original,velocity,noise,desired,t,mode):
    if original.ndim!=4 or original.shape[0]!=2 or velocity.shape!=original.shape or noise.shape!=original.shape:
        raise ValueError('Expected matching CFG latent, velocity and seeded noise')
    if not torch.equal(original[0],original[1]):raise ValueError('CFG input branches differ')
    if mode=='latent_target':return original.flip(-1)
    if mode=='latent_wrong_axis':return original.flip(-2)
    if mode not in ('seed_correlated_target','inferred_correlated_target','inferred_uncorrelated_target','seed_correlated_wrong_axis'):
        raise ValueError('Unknown reflection mode')
    residual=noise if mode.startswith('seed_') else original-float(t)*velocity
    if mode!='inferred_uncorrelated_target':
        residual=residual.flip(-2 if mode=='seed_correlated_wrong_axis' else -1)
    return linear_rebridge(residual,desired,t)


def continue_reflected(runtime,state,desired,mode):
    grid=state['time_grid'].to(runtime.device)
    if not torch.equal(grid,runtime.grid) or state['CFG']!=runtime.cfg:raise ValueError('Changed solver')
    if tensor_hash(state['prefix_cache'])!=state['prefix_cache_hash']:raise ValueError('Changed prefix')
    if state['runtime']['base_model_hash']!='a596cbc305c1df987c125d4f218e78f39b681621904cccfb2a3bf0ca0327f92c':raise ValueError('Changed base model')
    if runtime.config.transport.path_type!='Linear':raise ValueError('Requires Linear FM path')
    index=state['flow_index'];t=float(grid[index])
    if t!=state['true_t']:raise ValueError('Changed actual solver time')
    original=state['latent'].to(runtime.device).float()
    velocity=state['baseline_prediction'].to(runtime.device).float()
    noise=runtime.noise(state['seed']).float()
    x=reflected_input(original,velocity,noise,desired,t,mode)
    diagnostic=dict(mode=mode,source_input_sha256=tensor_hash(original),reflected_input_sha256=tensor_hash(x),
                    original_seed_noise_sha256=tensor_hash(noise),new_seed_chosen=False,
                    residual_reflected='uncorrelated' not in mode and 'correlated' in mode,prefix_cache_reused=False,
                    first_velocity_recomputed=True)
    runtime.model.diffusion_head_a=nn.ModuleList(runtime.original)
    kwargs=runtime.inputs(state['prompt']);calls=0
    for i in range(index,len(grid)-1):
        v=runtime.velocity(x,grid[i],kwargs)
        x=x+(grid[i+1]-grid[i])*v;calls+=1
    return x,dict(network_velocity_calls=calls,visual_block_calls=10*calls,shared_revisit_calls=0,mmu_checks=0,FLOPs=None),diagnostic
