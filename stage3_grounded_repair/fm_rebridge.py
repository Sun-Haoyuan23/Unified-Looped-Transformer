"""Offline Linear flow-matching bridge using the ORIGINAL seeded noise.

The local velocity-inferred residual x-t*v is generally not the original seed
noise after a nonlinear CFG trajectory. This causal control replaces that
residual with the original noise, not a newly chosen seed. Changed inputs always
recompute the FIRST full velocity; a cached prefix is never used on an edit.
"""
import torch
from torch import nn
from runtime import tensor_hash


def linear_rebridge(noise, clean, t):
    if noise.ndim != 4 or noise.shape[0] != 2 or clean.shape != noise[:1].shape:
        raise ValueError('Expected duplicated CFG noise and single-image clean')
    if not torch.equal(noise[0], noise[1]):
        raise ValueError('CFG noise branches differ')
    if not 0 <= float(t) <= 1:
        raise ValueError('Invalid flow time')
    value = (1 - t) * noise[:1].float() + t * clean.float()
    return torch.cat((value, value))


def continue_rebridged(runtime, state, desired, noise_source='seed'):
    grid = state['time_grid'].to(runtime.device)
    if not torch.equal(grid, runtime.grid) or state['CFG'] != runtime.cfg:
        raise ValueError('Captured flow grid/CFG differs')
    if tensor_hash(state['prefix_cache']) != state['prefix_cache_hash']:
        raise ValueError('Changed source prefix')
    if state['runtime']['base_model_hash'] != 'a596cbc305c1df987c125d4f218e78f39b681621904cccfb2a3bf0ca0327f92c':
        raise ValueError('Changed base model')
    if runtime.config.transport.path_type != 'Linear':
        raise ValueError('Bridge requires the official Linear flow matching path')
    index = state['flow_index']
    t = grid[index]
    if float(t) != state['true_t']:
        raise ValueError('Wrong actual shifted solver time')
    noise = runtime.noise(state['seed']).float()
    original = state['latent'].to(runtime.device).float()
    velocity = state['baseline_prediction'].to(runtime.device).float()
    inferred = original - t * velocity
    if noise_source not in ('seed', 'inferred'):
        raise ValueError('Unknown residual source')
    x = linear_rebridge(noise if noise_source == 'seed' else inferred, desired, t)
    diagnostic = {
        'original_seed_noise_sha256': tensor_hash(noise),
        'rebridged_input_sha256': tensor_hash(x),
        'inferred_vs_original_noise_relative_L2': float((inferred-noise).norm()/noise.norm()),
        'inferred_vs_original_noise_cosine': float(torch.nn.functional.cosine_similarity(inferred.flatten(), noise.flatten(), dim=0)),
        'rebridged_vs_saved_relative_L2': float((x-original).norm()/original.norm()),
        'first_velocity_recomputed': True, 'seed_changed': False, 'noise_source': noise_source}
    runtime.model.diffusion_head_a = nn.ModuleList(runtime.original)
    kwargs = runtime.inputs(state['prompt'])
    calls = 0
    for i in range(index, len(grid)-1):
        v = runtime.velocity(x, grid[i], kwargs)
        x = x + (grid[i+1]-grid[i])*v
        calls += 1
    return x, {'network_velocity_calls': calls, 'visual_block_calls': 10*calls,
               'shared_revisit_calls': 0, 'mmu_checks': 0, 'FLOPs': None}, diagnostic
