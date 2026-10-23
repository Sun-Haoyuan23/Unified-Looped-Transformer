"""Lightweight intermediate-loop readout for the shared visual-head loop."""

import torch
from torch import nn


class EarlyReadout(nn.Module):
    def __init__(self, hidden_size, width=4096):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, dtype=torch.float32)
        self.fc1 = nn.Linear(hidden_size, width, dtype=torch.float32)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(width, hidden_size, dtype=torch.float32)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, hidden):
        delta = self.fc2(self.act(self.fc1(self.norm(hidden.float()))))
        return (hidden.float() + delta).to(hidden.dtype)


class EarlyFinal(nn.Module):
    def __init__(self, final_head, readout):
        super().__init__()
        self.final_head = final_head
        self.readout = readout

    def forward(self, hidden, time_embeds, positions):
        return self.final_head(self.readout(hidden), time_embeds, positions)


def save_readout(model, path, steps, width, base_checkpoint):
    torch.save({"readout": {name: value.detach().to(torch.bfloat16).cpu()
                            for name, value in model.state_dict().items()},
                "steps": steps, "width": width,
                "base_checkpoint": base_checkpoint}, path)


def load_readout(model, path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["readout"], strict=True)
    return payload
