"""An additional, zero-gated visit to Show-o2's visual middle core.

The ten pretrained blocks are kept verbatim. This module revisits blocks 2–4
after block 7, while its zero gate leaves the pretrained prediction unchanged.
Only the gate is trainable in the small-data quality pilot, so any improvement
must arise from the second visit rather than new visual block weights.
"""

from pathlib import Path

import torch
from torch import nn


class GatedRevisit(nn.Module):
    def __init__(self, shared, hidden_size, max_gain=0.1, position=8):
        super().__init__()
        self.shared = nn.ModuleList(shared)
        self.gate = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.max_gain = float(max_gain)
        self.position = int(position)

    def forward(self, hidden_states, adaln_input, **kwargs):
        state = hidden_states
        # The visual core is frozen. Its second pass is a feature proposal;
        # gradients only update the gate and downstream operations.
        with torch.no_grad():
            for block in self.shared:
                state = block(hidden_states=state, adaln_input=adaln_input,
                              **kwargs)[0]
        gain = self.max_gain * self.gate.tanh()
        state = hidden_states + gain.to(hidden_states.dtype)[None, None, :] * (
            state - hidden_states)
        return (state,)


def insert_gated_revisit(model, max_gain=0.1, position=8):
    if len(model.diffusion_head_a) != 10:
        raise ValueError("Expected the original ten-block visual head")
    if position not in (5, 8):
        raise ValueError("Supported insertion positions are 5 and 8")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hidden_size = model.diffusion_head_config.hidden_size
    module = GatedRevisit(list(model.diffusion_head_a[2:5]), hidden_size,
                          max_gain=max_gain, position=position).to(device=next(model.parameters()).device)
    layers = list(model.diffusion_head_a)
    model.diffusion_head_a = nn.ModuleList(layers[:position] + [module] + layers[position:])
    model._elt_architecture = ("gated_revisit", 2, 3, position, 2, max_gain)
    return module


def save_gate(module, path: Path, steps: int):
    torch.save({"gate": module.gate.detach().cpu(), "steps": steps,
                "max_gain": module.max_gain, "position": module.position}, path)


def load_gate(module, path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["max_gain"] != module.max_gain:
        raise ValueError("Gate gain mismatch")
    if payload.get("position", 8) != module.position:
        raise ValueError("Gate position mismatch")
    with torch.no_grad():
        module.gate.copy_(payload["gate"].to(module.gate.device))
    return payload
