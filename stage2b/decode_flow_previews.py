"""Decode one-step clean estimates from saved on-policy Show-o2 flow states."""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", default=[0.45, 0.65, 0.85, 0.95])
    args = parser.parse_args()
    sys.path.insert(0, str(args.work / "Show-o" / "show-o2"))
    from models import WanVAE
    from utils import denorm

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    vae = WanVAE(vae_pth="/home/shy_23/models/show-o2-deps/Wan2.1_VAE.pth",
                 dtype=dtype, device=device)
    tasks = json.loads(args.manifest.read_text(encoding="utf-8"))["val"]
    rows = []
    with torch.inference_mode():
        for task in tasks:
            task_id = task["id"]
            payload = torch.load(args.work / "stage2b" / "trajectories" / "val" /
                                 f"{task_id}.pt", map_location="cpu", weights_only=False)
            t_values = payload["t"].float()
            for requested in args.times:
                index = int(torch.argmin((t_values - requested).abs()))
                actual = float(t_values[index])
                latent = (payload["zt"][index].float() +
                          (1 - actual) * payload["teacher_velocity"][index].float())
                pixels = vae.batch_decode(latent.to(device=device, dtype=dtype).unsqueeze(2)).squeeze(2)
                tag = f"t{round(requested * 100):02d}"
                directory = args.output / tag / task_id
                (directory / "samples").mkdir(parents=True, exist_ok=True)
                (directory / "metadata.jsonl").write_text(
                    (args.source_layout / task_id / "metadata.jsonl").read_text(encoding="utf-8"),
                    encoding="utf-8")
                path = directory / "samples" / "0000.png"
                Image.fromarray(denorm(pixels)[0]).save(path)
                row = {"id": task_id, "category": task["category"],
                       "requested_t": requested, "actual_t": actual,
                       "trajectory_index": index, "image": str(path)}
                rows.append(row)
                print(json.dumps(row), flush=True)
    (args.output / "index.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
