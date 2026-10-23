"""Compare each reconstructed loop projection with its pretrained source."""

import argparse
import json
import sys
from pathlib import Path

import torch

from elt_model import install_svd_middle_head, load_head_checkpoint, DeltaLoRALinear


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.work / "Show-o" / "show-o2"))
    from models import Showo2Qwen2_5
    model = Showo2Qwen2_5.from_pretrained(
        str(args.work / "models" / "show-o2-1.5B-local"),
        use_safetensors=False).to(device="cuda:0", dtype=torch.bfloat16).eval()
    targets = list(model.diffusion_head_a)[5:8]
    install_svd_middle_head(model, rank=args.rank, trainable=False)
    load_head_checkpoint(model, args.work / "stage2b" / args.checkpoint)
    all_rows = []
    for index, (adapted, target) in enumerate(zip(
            list(model.diffusion_head_a)[5:8], targets)):
        for name, module in adapted.block.named_modules():
            if not isinstance(module, DeltaLoRALinear):
                continue
            target_module = dict(target.named_modules())[name]
            composed = (module.base.weight.float() +
                        (module.b @ module.a)*module.scale)
            diff = composed - target_module.weight.float()
            target_delta = target_module.weight.float() - module.base.weight.float()
            reconstructed_delta = (module.b @ module.a)*module.scale
            row = {"block": index+5, "module": name,
                   "rank": module.a.shape[0],
                   "weight_mse": float(diff.square().mean()),
                   "relative_fro": float(diff.norm()/target_module.weight.float().norm()),
                   "target_delta_norm": float(target_delta.norm()),
                   "reconstructed_delta_norm": float(reconstructed_delta.norm()),
                   "relative_delta_error": float(diff.norm()/target_delta.norm()),
                   "delta_cosine": float(torch.nn.functional.cosine_similarity(
                       target_delta.flatten()[None],
                       reconstructed_delta.flatten()[None]).item()),
                   "max_abs": float(diff.abs().max()),
                   "cast_equal_fraction": float((composed.to(torch.bfloat16) ==
                                                  target_module.weight).float().mean())}
            all_rows.append(row)
            print(json.dumps(row), flush=True)
    print(json.dumps({"max_relative_fro": max(r["relative_fro"] for r in all_rows),
                      "max_weight_mse": max(r["weight_mse"] for r in all_rows)}),
          flush=True)


if __name__ == "__main__":
    main()
