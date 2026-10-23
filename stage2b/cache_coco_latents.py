"""Encode downloaded COCO train JPEGs with the released Show-o2 WanVAE."""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image


def image_transform(image, resolution=432):
    """Show-o2 Resize(shorter side)+CenterCrop+[-1,1], without torchvision."""
    width, height = image.size
    if width < height:
        size = (resolution, round(height * resolution / width))
    else:
        size = (round(width * resolution / height), resolution)
    image = image.resize(size, Image.Resampling.BICUBIC)
    left = (image.width - resolution) // 2
    top = (image.height - resolution) // 2
    image = image.crop((left, top, left + resolution, top + resolution))
    pixels = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one healthy GPU by UUID")
    sys.path.insert(0, str(args.work / "Show-o" / "show-o2"))
    from models import WanVAE

    device = torch.device("cuda:0")
    vae = WanVAE(vae_pth="/home/shy_23/models/show-o2-deps/Wan2.1_VAE.pth",
                 dtype=torch.bfloat16, device=device)
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    if args.max_records:
        records = records[:args.max_records]
    latent_dir = args.data / "latents"
    latent_dir.mkdir(exist_ok=True)
    done = 0
    with torch.inference_mode():
        for row in records:
            target = latent_dir / f"{row['id']:012d}.pt"
            if target.exists():
                done += 1
                continue
            with Image.open(args.data / "images" / row["file_name"]) as image:
                pixels = image_transform(image.convert("RGB"), resolution=432)
            z = vae.sample(pixels.unsqueeze(0).unsqueeze(2).to(device),
                           deterministic=True).squeeze(2)
            torch.save({"id": row["id"], "z": z.cpu().to(torch.bfloat16),
                        "prompt": row["prompt"], "split": row["split"]}, target)
            done += 1
            if done % 50 == 0 or done == len(records):
                print(json.dumps({"done": done, "total": len(records),
                                  "latent_shape": list(z.shape)}), flush=True)


if __name__ == "__main__":
    main()
