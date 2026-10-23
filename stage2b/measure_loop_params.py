"""Count distinct trainable/storage parameters in pretrained and looped Show-o2."""

import argparse
import json
import sys
from pathlib import Path

import torch

from elt_model import install_svd_loop_head


def count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=1024)
    args = parser.parse_args()
    source = args.work / "Show-o" / "show-o2"
    sys.path.insert(0, str(source))
    from models import Showo2Qwen2_5

    model = Showo2Qwen2_5.from_pretrained(
        str(args.work / "models" / "show-o2-1.5B-local"),
        use_safetensors=False).to(device="cuda:0", dtype=torch.bfloat16).eval()
    baseline = {"total": count(model),
                "visual_blocks": count(model.diffusion_head_a),
                "last_five_visual_blocks": sum(count(block)
                                               for block in list(model.diffusion_head_a)[5:]),
                "visual_final": count(model.diffusion_head_b)}
    adapters = install_svd_loop_head(model, rank=args.rank,
                                     svd_init=False, trainable=True)
    looped = {"total": count(model),
              "visual_blocks": count(model.diffusion_head_a),
              "visual_final": count(model.diffusion_head_b),
              "adapter_parameters": sum(parameter.numel()
                                        for parameter in adapters)}
    result = {"rank": args.rank, "baseline": baseline, "looped": looped,
              "total_parameter_reduction": baseline["total"] - looped["total"],
              "visual_block_reduction": (baseline["visual_blocks"] -
                                         looped["visual_blocks"])}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
