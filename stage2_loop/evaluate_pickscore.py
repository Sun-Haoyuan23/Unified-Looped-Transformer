"""Independent PickScore proxy for paired baseline versus ELT loop exits."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--variants", default="base,L1,L2,L3")
    parser.add_argument("--image-dir", default="elt_images")
    parser.add_argument("--tag", default="elt")
    args = parser.parse_args()
    variants = args.variants.split(",")
    if not variants or variants[0] != "base":
        raise ValueError("First variant must be base")
    tasks = json.loads((args.work / "prompts.json").read_text(encoding="utf-8"))
    processor = AutoProcessor.from_pretrained(args.work / "models" / "pickscore-processor")
    model = AutoModel.from_pretrained(args.work / "models" / "pickscore-v1",
                                      use_safetensors=True).eval().to(args.device)
    text_features = {}
    with torch.inference_mode():
        for task in tasks:
            encoded = processor(text=[task["prompt"]], padding=True, truncation=True,
                                max_length=77, return_tensors="pt").to(args.device)
            vec = model.get_text_features(**encoded).float()
            text_features[task["id"]] = torch.nn.functional.normalize(vec, dim=-1)[0].cpu()
    items = []
    for task in tasks:
        for seed in task["seeds"]:
            for variant in variants:
                path = ((args.work / "images" / f"{task['id']}_seed{seed}.png")
                        if variant == "base" else
                        (args.work / "stage2_loop" / args.image_dir /
                         f"{task['id']}_seed{seed}_{variant}.png"))
                if not path.exists():
                    raise FileNotFoundError(path)
                items.append((task, seed, variant, path))
    rows = []
    with torch.inference_mode():
        for offset in range(0, len(items), args.batch_size):
            batch = items[offset:offset + args.batch_size]
            images = [Image.open(item[3]).convert("RGB") for item in batch]
            encoded = processor(images=images, padding=True, truncation=True,
                                max_length=77, return_tensors="pt").to(args.device)
            vec = model.get_image_features(**encoded).float()
            vec = torch.nn.functional.normalize(vec, dim=-1).cpu()
            scale = float(model.logit_scale.exp().detach().cpu())
            for (task, seed, variant, _path), image_vec in zip(batch, vec):
                score = scale * float(torch.dot(text_features[task["id"]], image_vec))
                rows.append({"id": task["id"], "category": task["category"],
                             "seed": seed, "variant": variant,
                             "pickscore_logit": round(score, 6)})
            print(f"scored {min(offset+args.batch_size, len(items))}/{len(items)}", flush=True)
    root = args.work / "stage2_loop"
    with (root / f"{args.tag}_pickscore.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lookup = {(row["id"], row["seed"], row["variant"]): row["pickscore_logit"]
              for row in rows}
    summary = {"note": "PickScore is a preference proxy, not human votes",
               "model_repo": "yuvalkirstain/PickScore_v1",
               "pairs_per_variant": len(tasks) * len(tasks[0]["seeds"]),
               "variants": {}}
    for variant in variants:
        values = [row["pickscore_logit"] for row in rows if row["variant"] == variant]
        stats = {"mean": round(statistics.mean(values), 6),
                 "median": round(statistics.median(values), 6)}
        if variant != "base":
            deltas = [lookup[(task["id"], seed, variant)] -
                      lookup[(task["id"], seed, "base")]
                      for task in tasks for seed in task["seeds"]]
            stats["mean_paired_change"] = round(statistics.mean(deltas), 6)
            stats["fraction_higher"] = round(sum(d > 0 for d in deltas) / len(deltas), 4)
        summary["variants"][variant] = stats
    (root / f"{args.tag}_pickscore_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                 encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
