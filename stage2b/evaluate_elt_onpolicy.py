"""Paired SigLIP alignment/diversity and blinded visual review for whole-head loop."""

import argparse
import csv
import json
import random
import statistics
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, SiglipModel


def draw_sheet(tasks, image_paths, variants, output):
    cell_w, cell_h = 216, 242
    sheet = Image.new("RGB", (cell_w * len(variants), cell_h * len(tasks)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, task in enumerate(tasks):
        seed = task["seeds"][0]
        for col, variant in enumerate(variants):
            path = image_paths[(task["id"], seed, variant)]
            image = Image.open(path).convert("RGB").resize((216, 216))
            sheet.paste(image, (cell_w * col, cell_h * row + 26))
            draw.text((cell_w * col + 5, cell_h * row + 5),
                      f"{task['id']} | {variant}", fill="black")
    sheet.save(output, quality=92)


def blind_cards(tasks, image_paths, variants, root, tag):
    rng = random.Random(20261003)
    rows = []
    cards = root / f"{tag}_blind_pairs"
    cards.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        seed = task["seeds"][0]
        for variant in variants[1:]:
            left, right = ("base", variant) if rng.random() < 0.5 else (variant, "base")
            canvas = Image.new("RGB", (864, 482), "white")
            for col, method in enumerate((left, right)):
                image = Image.open(image_paths[(task["id"], seed, method)]).convert("RGB")
                canvas.paste(image, (432 * col, 50))
            ImageDraw.Draw(canvas).text((8, 10),
                                        f"{task['id']} seed {seed} | A (left) vs B (right)",
                                        fill="black")
            card = f"{task['id']}_seed{seed}_{variant}.jpg"
            canvas.save(cards / card, quality=92)
            rows.append({"card": card, "prompt": task["prompt"], "A": left, "B": right,
                         "winner_A_B_tie": "", "quality_A_1to5": "", "quality_B_1to5": "",
                         "prompt_following_A_1to5": "", "prompt_following_B_1to5": "",
                         "comments": ""})
    with (root / f"{tag}_blind_review.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "A": "", "B": ""})
    with (root / f"{tag}_blind_key.json").open("w", encoding="utf-8") as file:
        json.dump([{k: row[k] for k in ("card", "A", "B")} for row in rows], file, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--variants", default="base,L1,L2")
    parser.add_argument("--image-dir", default="elt_images")
    parser.add_argument("--tag", default="elt_onpolicy")
    parser.add_argument("--tasks-file", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    args = parser.parse_args()
    root = args.work / "stage2b"
    tasks = json.loads((args.tasks_file or args.work / "prompts.json").read_text(
        encoding="utf-8"))
    baseline_dir = args.baseline_dir or args.work / "images"
    variants = args.variants.split(",")
    if not variants or variants[0] != "base":
        raise ValueError("First variant must be base")
    image_paths = {}
    for task in tasks:
        for seed in task["seeds"]:
            for variant in variants:
                path = ((baseline_dir / f"{task['id']}_seed{seed}.png")
                        if variant == "base" else
                        (root / args.image_dir / f"{task['id']}_seed{seed}_{variant}.png"))
                if not path.exists():
                    raise FileNotFoundError(path)
                if Image.open(path).size != (432, 432):
                    raise RuntimeError(f"Unexpected dimensions: {path}")
                image_paths[(task["id"], seed, variant)] = path

    model_path = args.work / "models" / "siglip-so400m"
    processor = AutoProcessor.from_pretrained(model_path)
    model = SiglipModel.from_pretrained(model_path).to(args.device).eval()
    text_features = {}
    with torch.inference_mode():
        for task in tasks:
            encoded = processor(text=[task["prompt"]], padding="max_length",
                                return_tensors="pt")
            encoded = {k: v.to(args.device) for k, v in encoded.items()}
            vec = model.get_text_features(**encoded).float()
            text_features[task["id"]] = torch.nn.functional.normalize(vec, dim=-1)[0].cpu()
    keys = list(image_paths)
    features = {}
    with torch.inference_mode():
        for offset in range(0, len(keys), args.batch_size):
            batch_keys = keys[offset:offset + args.batch_size]
            images = [Image.open(image_paths[key]).convert("RGB") for key in batch_keys]
            encoded = processor(images=images, return_tensors="pt")
            pixel_values = encoded["pixel_values"].to(args.device)
            vec = model.get_image_features(pixel_values=pixel_values).float()
            vec = torch.nn.functional.normalize(vec, dim=-1).cpu()
            for key, feature in zip(batch_keys, vec):
                features[key] = feature.numpy()
            print(f"embedded {min(offset + args.batch_size, len(keys))}/{len(keys)}", flush=True)
    rows = []
    for task in tasks:
        for seed in task["seeds"]:
            for variant in variants:
                key = (task["id"], seed, variant)
                score = float(np.dot(features[key], text_features[task["id"]].numpy()))
                rows.append({"id": task["id"], "category": task["category"],
                             "seed": seed, "variant": variant,
                             "siglip_text_image_cosine": round(score, 6)})
    with (root / f"{args.tag}_alignment.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_key = {(row["id"], row["seed"], row["variant"]): row for row in rows}
    summary = {"image_count": len(keys), "prompt_count": len(tasks),
               "seeds_per_prompt": [len(task["seeds"]) for task in tasks],
               "variants": {}}
    for variant in variants:
        scores = [row["siglip_text_image_cosine"] for row in rows
                  if row["variant"] == variant]
        distances = []
        for task in tasks:
            group = [features[(task["id"], seed, variant)] for seed in task["seeds"]]
            if len(group) > 1:
                distances.append(statistics.mean(
                    float(1 - np.dot(a, b)) for a, b in combinations(group, 2)))
        result = {"mean_siglip_alignment": round(statistics.mean(scores), 6),
                  "mean_within_prompt_siglip_distance": (round(statistics.mean(distances), 6)
                                                         if distances else None),
                  "median_within_prompt_siglip_distance": (round(statistics.median(distances), 6)
                                                           if distances else None)}
        if variant != "base":
            deltas = [by_key[(task["id"], seed, variant)]["siglip_text_image_cosine"] -
                      by_key[(task["id"], seed, "base")]["siglip_text_image_cosine"]
                      for task in tasks for seed in task["seeds"]]
            result["mean_paired_alignment_change"] = round(statistics.mean(deltas), 6)
            result["fraction_alignment_higher"] = round(sum(x > 0 for x in deltas)/len(deltas), 4)
        summary["variants"][variant] = result
    (root / f"{args.tag}_eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                             encoding="utf-8")
    draw_sheet(tasks, image_paths, variants, root / f"{args.tag}_contact_sheet.jpg")
    blind_cards(tasks, image_paths, variants, root, args.tag)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
