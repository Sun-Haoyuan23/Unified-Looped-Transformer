"""TRAIN-only native trajectory controls. No learned operator or controller."""
import torch

ACTIONS = ('baseline', 'checked_zero', 'source_final_rebridge',
           'native_parent_direct', 'native_parent_additive', 'native_final_rebridge')


def physical(value):
    if value.ndim != 4 or value.shape[0] != 2 or value.dtype != torch.float32:
        raise ValueError('Require duplicated FP32 CFG physical state')
    if not torch.isfinite(value).all() or not torch.equal(value[0], value[1]):
        raise ValueError('Nonfinite or unequal CFG physical branches')
    return value


def validate_identity(source, parent, pair):
    for state, field, seed_key in ((source, 'source_state_id', 'source_seed'),
                                  (parent, 'native_parent_state_id', 'parent_seed')):
        if state['state_id'] != pair[field] or state['seed'] != pair[seed_key]:
            raise ValueError('State identity mismatch')
        for key in ('split', 'prompt_id', 'prompt', 'flow_index', 'true_t'):
            if state[key] != pair[key]:
                raise ValueError('Pair metadata differs: ' + key)
        if state['split'] != 'train' or state['flow_index'] != 35:
            raise ValueError('Only frozen TRAIN index35 is allowed')
        physical(state['latent'])
        physical(state['baseline_final_latent'])
        if float(state['time_grid'][35]) != state['true_t']:
            raise ValueError('Actual solver time differs')
    if source['seed'] == parent['seed'] or source['CFG'] != parent['CFG']:
        raise ValueError('Expected matched CFG and different seeds')
    if not torch.equal(source['time_grid'], parent['time_grid']):
        raise ValueError('Parent and source solver grids differ')
    if source['runtime']['base_model_hash'] != parent['runtime']['base_model_hash']:
        raise ValueError('Base model differs')


def inferred_noise(original, velocity, t):
    physical(original); physical(velocity)
    if t.ndim != 0 or t.dtype != torch.float32 or not 0 < float(t) < 1:
        raise ValueError('Require original scalar FP32 solver time')
    return physical(original - t * velocity)


def rebridge(noise, final, t):
    physical(noise); physical(final)
    # Preserve original grid-tensor arithmetic; do not use Python float(t).
    return physical((1 - t) * noise + t * final)


def additive_parent(source, parent):
    physical(source); physical(parent)
    delta = parent - source
    return physical(source + delta), physical(delta)
