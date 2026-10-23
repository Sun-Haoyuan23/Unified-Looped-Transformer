"""Matched nonlinear readout of real recurrent or pre-recurrence visual tokens.

No state IDs, teachers, masks, images, terminal labels or halting decisions enter
this mapping. A fixed repair invocation predicts one CFG-shared physical delta.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class NonlinearTokenReadout(nn.Module):
    def __init__(self, input_rms, outputs=64, width=128):
        super().__init__()
        if input_rms.ndim != 1 or not torch.isfinite(input_rms).all() or (input_rms <= 0).any():
            raise ValueError('Finite positive TRAIN feature RMS required')
        self.register_buffer('input_rms', input_rms.detach().clone().float())
        self.input = nn.Linear(input_rms.numel(), width)
        self.output = nn.Linear(width, outputs)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, tokens):
        if tokens.dtype != torch.float32 or tokens.shape[-1] != self.input_rms.numel():
            raise ValueError('Expected FP32 aligned token features')
        return self.output(F.silu(self.input(tokens / self.input_rms)))


def joint_token_features(operator, features, positions, mode='loop2'):
    """Differentiable joint features; no old physical-state/KV cache continuation."""
    if mode not in ('loop2', 'no_loop'):
        raise ValueError('Only two frozen matched routes are supported')
    values = []
    calls = {'outer': 0, 'inner': 0}
    for name in ('outer', 'inner'):
        branch = getattr(operator, name)
        if branch is None:
            raise ValueError('Joint outer and inner branches are required')
        hidden, kwargs = features[name]
        if kwargs.get('use_cache', False) or kwargs.get('past_key_value') is not None:
            raise ValueError('Shared recurrence must not reuse KV')
        offset, length = positions[0, 0].tolist()
        begin, end = offset + 1, offset + length
        count = length - 1
        side = math.isqrt(count)
        if hidden.shape[0] != 2 or count <= 0 or side * side != count or end > hidden.shape[1]:
            raise ValueError('Invalid conditional visual token span')
        line = torch.linspace(-1, 1, side, device=hidden.device)
        yy, xx = torch.meshgrid(line, line, indexing='ij')
        coords = torch.stack((xx, yy), -1).reshape(1, count, 2)
        field = F.silu(branch.visual_project(branch.norm(hidden[:1, begin:end]).float()) +
                       branch.encoder(operator.condition)[:, None] + branch.position_project(coords))
        delta = torch.zeros_like(hidden)
        delta[0, begin:end] = branch.feedback(field)[0].to(hidden.dtype)
        h = hidden + delta
        clean = {k: v for k, v in kwargs.items() if k != 'hidden_states'}
        if mode == 'loop2':
            for _ in range(2):
                for index in range(len(branch.shared_core)):
                    h = branch.run_core(index, h, clean)
                    calls[name] += 1
        values.append(torch.cat((branch.norm(h[:1, begin:end]).float(), field, coords), -1))
    return torch.cat(values, -1), calls


def patch_delta(runtime, patches, latent):
    if patches.ndim != 3 or patches.shape[0] != 1:
        raise ValueError('Readout must predict one conditional patch sequence')
    decoded = runtime.model.unpatchify(patches, latent.shape[-2] // runtime.patch,
                                      latent.shape[-1] // runtime.patch)
    value = decoded.transpose(1, 2).reshape(1, *latent.shape[1:]).float()
    return torch.cat((value, value), 0)
