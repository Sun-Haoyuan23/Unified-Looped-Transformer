"""A shared ten-block visual transformer run twice per flow evaluation.

At zero adapter weights, the two-loop output is exactly the released Show-o2
ten-block prediction. This gives a quality-preserving initialization while
testing whether a second representation pass replaces flow solver steps.
"""

import torch
from torch import nn


class StoreExit(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.hidden = None

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.hidden = output[0]
        return output


class RefineExit(nn.Module):
    def __init__(self, block, store, hidden_size, width):
        super().__init__()
        self.block = block
        self.store = store
        self.adapter = nn.Sequential(
            nn.LayerNorm(2 * hidden_size),
            nn.Linear(2 * hidden_size, width),
            nn.SiLU(),
            nn.Linear(width, hidden_size),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, *args, **kwargs):
        second = self.block(*args, **kwargs)
        first_hidden = self.store.hidden
        if first_hidden is None:
            raise RuntimeError("Missing first-loop visual representation")
        first = first_hidden.float()
        delta = second[0].float() - first
        with torch.autocast("cuda", enabled=False):
            correction = self.adapter(torch.cat((first, delta), dim=-1))
        refined = (first + correction).to(second[0].dtype)
        return (refined,) + second[1:]


def install_loop_solver(model, width=128, trainable=True):
    blocks = list(model.diffusion_head_a)
    if len(blocks) != 10:
        raise ValueError("Expected ten original Show-o2 visual blocks")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    store = StoreExit(blocks[-1])
    refine = RefineExit(blocks[-1], store,
                        model.diffusion_head_config.hidden_size, width)
    refine.adapter.to(device=next(model.parameters()).device,
                      dtype=torch.float32)
    # The same ten pretrained Transformer blocks execute in both passes.
    model.diffusion_head_a = nn.ModuleList(blocks[:-1] + [store] +
                                           blocks[:-1] + [refine])
    if trainable:
        for parameter in refine.adapter.parameters():
            parameter.requires_grad_(True)
    return refine


def save_loop_adapter(refine, path, steps, width):
    state = {name: value.detach().to(torch.bfloat16).cpu()
             for name, value in refine.adapter.state_dict().items()}
    torch.save({"adapter": state, "steps": steps, "width": width,
                "architecture": "10 shared visual blocks x 2 loops + readout adapter"},
               path)


def load_loop_adapter(refine, path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["width"] != refine.adapter[1].out_features:
        raise RuntimeError("Adapter width mismatch")
    refine.adapter.load_state_dict(payload["adapter"])
    return payload
