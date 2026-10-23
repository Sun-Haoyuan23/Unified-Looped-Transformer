"""Cheap time-conditioned approximation of one recurrent visual-block visit."""

import torch
from torch import nn


class EarlyBridge(nn.Module):
    """Map the first shared-block state toward its second-visit state.

    This is used only by the short exit. The full exit still executes the
    genuinely shared recurrent block. Its last projection starts at zero, so
    installing an untrained bridge leaves the existing short exit unchanged.
    """

    def __init__(self, hidden_size, width=512):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, width)
        self.time = nn.Linear(hidden_size, width, bias=False)
        self.act = nn.SiLU()
        self.up = nn.Linear(width, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states, adaln_input, **_kwargs):
        dtype = hidden_states.dtype
        hidden = hidden_states.float()
        time = adaln_input.float()
        update = self.up(self.act(self.down(self.norm(hidden)) +
                                  self.time(time)[:, None, :]))
        return ((hidden + update).to(dtype),)


def load_bridge(module, path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    module.load_state_dict(payload["bridge"], strict=True)
    return payload
