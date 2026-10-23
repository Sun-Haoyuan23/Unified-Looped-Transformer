"""Tune only Show-o2's shared visual prediction head after loop compression."""

import torch

from elt_model import DtypeFinal


def install_tunable_final(model):
    model.diffusion_head_b = DtypeFinal(model.diffusion_head_b.to(torch.float32))
    return list(model.diffusion_head_b.parameters())


def save_final(model, path, steps, loop_checkpoint):
    torch.save({"final_head": {name: value.detach().to(torch.bfloat16).cpu()
                               for name, value in model.diffusion_head_b.state_dict().items()},
                "steps": steps, "loop_checkpoint": loop_checkpoint}, path)


def load_final(model, path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    install_tunable_final(model)
    model.diffusion_head_b.load_state_dict(payload["final_head"], strict=True)
    return payload
