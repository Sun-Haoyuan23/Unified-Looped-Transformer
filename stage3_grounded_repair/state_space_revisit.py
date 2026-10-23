"""Evidence-conditioned shared recurrence reads out a physical latent-state update.

The original outer/inner groups execute first. The exact same frozen groups are
then revisited once; their token states feed a zero-initialized latent readout.
No image edits, object masks, teacher tensors or final labels enter this forward.
"""
from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.nn import functional as F
from evidence_encoder import EvidenceEncoder


class StateBranch(nn.Module):
    def __init__(self, cores, hidden_size, patch_features, encoder, width=64):
        super().__init__()
        self.shared_core = nn.ModuleList(cores)
        self.encoder = encoder
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.visual_project = nn.Linear(hidden_size, width, bias=False)
        self.position_project = nn.Linear(2, width, bias=False)
        self.feedback = nn.Linear(width, hidden_size, bias=False)
        self.readout = nn.Linear(hidden_size + width + 2, patch_features)
        nn.init.zeros_(self.feedback.weight)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        for p in self.shared_core.parameters():
            p.requires_grad_(False)

    def forward(self, hidden, kwargs, condition, positions, revisit=True):
        if kwargs.get('use_cache', False) or kwargs.get('past_key_value') is not None:
            raise ValueError('State recurrence requires cache-free computation')
        offset, length = positions[0, 0].tolist()
        count = length - 1
        side = math.isqrt(count)
        begin, end = offset + 1, offset + length
        if hidden.shape[0] != 2 or count <= 0 or side * side != count or end > hidden.shape[1]:
            raise ValueError('Expected a square conditional visual span and B2 CFG')
        line = torch.linspace(-1, 1, side, device=hidden.device)
        yy, xx = torch.meshgrid(line, line, indexing='ij')
        coords = torch.stack((xx, yy), -1).reshape(1, count, 2)
        field = F.silu(self.visual_project(self.norm(hidden[:1, begin:end]).float()) +
                       self.encoder(condition)[:, None] + self.position_project(coords))
        # Only conditional visual tokens receive feedback. The physical latent
        # readout is later duplicated for both CFG branches, rather than creating
        # two different images for conditional and unconditional trajectories.
        delta = torch.zeros_like(hidden)
        delta[0, begin:end] = self.feedback(field)[0].to(hidden.dtype)
        h = hidden + delta
        clean_kwargs = {k: v for k, v in kwargs.items() if k != 'hidden_states'}
        if revisit:
            for core in self.shared_core:
                h = core(hidden_states=h, **clean_kwargs)[0]
        patches = self.readout(torch.cat((self.norm(h[:1, begin:end]).float(), field, coords), -1))
        return patches


class StateSpaceRevisit(nn.Module):
    def __init__(self, outer, inner, arm):
        super().__init__()
        self.outer, self.inner, self.arm = outer, inner, arm
        self.condition = None

    @property
    def encoder(self):
        return (self.outer if self.outer is not None else self.inner).encoder

    def trainable_parameters(self):
        return [p for name, p in self.named_parameters() if '.shared_core.' not in name]

    def adapter_state(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if '.shared_core.' not in k}

    def core_hashes(self):
        from runtime import tensor_hash
        return {name: tensor_hash(t) for name, t in self.state_dict().items() if '.shared_core.' in name}

    def latent_delta(self, runtime, features, positions, latent):
        values = []
        for name, branch in (('outer', self.outer), ('inner', self.inner)):
            if branch is None:
                continue
            h, kwargs = features[name]
            patches = branch(h, kwargs, self.condition, positions, revisit=self.arm != 'head_only')
            decoded = runtime.model.unpatchify(patches, latent.shape[-2] // runtime.patch,
                                              latent.shape[-1] // runtime.patch)
            values.append(decoded.transpose(1, 2).reshape(1, *latent.shape[1:]).float())
        delta = torch.stack(values).sum(0)
        return torch.cat((delta, delta), 0)


def make_operator(runtime, arm, level=2, width=64):
    if arm not in ('head_only', 'outer', 'inner', 'joint'):
        raise ValueError('Unknown state-space arm')
    layers = runtime.model.showo.model.layers
    encoder = EvidenceEncoder(level=level, width=width)
    patch_features = runtime.channels * runtime.patch ** 2
    # Always build both branches in this order so matching arms have exactly the
    # same initialization despite different module subsets.
    outer = StateBranch(list(layers[-2:]), runtime.model.showo.config.hidden_size,
                        patch_features, encoder, width)
    inner = StateBranch(list(runtime.original[-2:]), runtime.model.diffusion_head_config.hidden_size,
                        patch_features, encoder, width)
    return StateSpaceRevisit(outer if arm != 'inner' else None,
                             inner if arm in ('inner', 'joint') else None, arm).to(runtime.device)


@contextmanager
def full_model_context(runtime):
    config = runtime.model.showo.config
    previous_cache = config.use_cache
    previous_head = runtime.model.diffusion_head_a
    try:
        config.use_cache = False
        runtime.model.diffusion_head_a = nn.ModuleList(runtime.original)
        yield
    finally:
        runtime.model.diffusion_head_a = previous_head
        config.use_cache = previous_cache


def capture_features(runtime, state, operator):
    """Read features before feedback. No captured inner prefix or old KV is used."""
    from runtime import rng_restore, tensor_hash
    if tensor_hash(state['latent']) != state['state_fingerprint']:
        raise ValueError('Captured state corrupted')
    if not torch.equal(state['time_grid'].to(runtime.device), runtime.grid) or state['CFG'] != runtime.cfg:
        raise ValueError('Solver/CFG mismatch')
    options = runtime.inputs(state['prompt'])
    features, hooks = {}, []

    def hook(name):
        def save(module, args, kwargs, output):
            if name in features:
                raise ValueError('Expected one original visit during feature capture')
            context = {k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                       for k, v in kwargs.items() if k != 'hidden_states'}
            if args and len(args) > 1:
                raise ValueError('Unsupported positional decoder context')
            features[name] = (output[0].detach().clone(), context)
        return save

    try:
        if operator.outer is not None:
            hooks.append(runtime.model.showo.model.layers[-1].register_forward_hook(hook('outer'), with_kwargs=True))
        if operator.inner is not None:
            hooks.append(runtime.original[-1].register_forward_hook(hook('inner'), with_kwargs=True))
        rng_restore(state['rng'])
        with full_model_context(runtime), torch.no_grad():
            runtime.velocity(state['latent'].to(runtime.device), runtime.grid[state['flow_index']], options)
    finally:
        for handle in hooks:
            handle.remove()
    required = {'outer'} if operator.arm in ('outer', 'head_only') else {'inner'} if operator.arm == 'inner' else {'outer', 'inner'}
    if set(features) != required:
        raise ValueError('Missing original-group features')
    features['_provenance'] = (state['state_fingerprint'], state['prompt'], state['flow_index'])
    return features, options


def replay_state(runtime, state, operator, features=None, checkpoint_rollout=False, apply_update=True):
    """One learned state update, followed by all original remaining flow steps."""
    from torch.utils.checkpoint import checkpoint
    from runtime import rng_restore, tensor_hash
    if tensor_hash(state['latent']) != state['state_fingerprint']:
        raise ValueError('Changed latent state')
    if not torch.equal(state['time_grid'].to(runtime.device), runtime.grid) or state['CFG'] != runtime.cfg:
        raise ValueError('Solver/CFG mismatch')
    if apply_update and features is None:
        features, options = capture_features(runtime, state, operator)
    else:
        # A cache is only of PRE-feedback features for this unmodified state;
        # every velocity after the latent update remains a full fresh forward.
        options = runtime.inputs(state['prompt'])
    x = state['latent'].to(runtime.device).clone()
    if apply_update:
        if features['_provenance'] != (state['state_fingerprint'], state['prompt'], state['flow_index']):
            raise ValueError('Pre-feedback cache belongs to a different state')
        x = x + operator.latent_delta(runtime, features, options['modality_positions'], x)
    grid = state['time_grid'].to(runtime.device)
    rng_restore(state['rng'])
    calls = 0
    for i in range(state['flow_index'], len(grid) - 1):
        def velocity(z, now=grid[i]):
            return runtime.velocity(z, now, options)
        v = checkpoint(velocity, x, use_reentrant=False) if checkpoint_rollout else velocity(x)
        x = x + (grid[i + 1] - grid[i]) * v
        calls += 1
    extra_outer = 2 if apply_update and operator.outer is not None and operator.arm != 'head_only' else 0
    extra_inner = 2 if apply_update and operator.inner is not None else 0
    capture_calls = int(apply_update)
    return x, dict(network_velocity_calls=calls + capture_calls, post_update_velocity_calls=calls,
                   feature_capture_velocity_calls=capture_calls, outer_backbone_layer_calls=(calls + capture_calls) * 28,
                   flow_head_layer_calls=(calls + capture_calls) * 10, semantic_encoder_layer_calls_nominal=(calls + capture_calls) * 26,
                   outer_shared_revisit_calls=extra_outer, inner_shared_revisit_calls=extra_inner,
                   nominal_full_network_velocity_calls=49 + capture_calls,
                   FLOPs=None)
