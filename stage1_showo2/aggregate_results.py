"""Validate all stage-1 generations and summarize reproducible costs."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from PIL import Image, ImageDraw


def median(values):
    return round(statistics.median(values), 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    work = args.work
    tasks = json.loads((work / "prompts.json").read_text(encoding="utf-8"))
    expected = {(task["id"], seed) for task in tasks for seed in task["seeds"]}
    rows = [json.loads(line) for gpu in (0, 1, 2, 3, 4, 6, 7)
            for line in (work / "logs" / f"metrics_gpu{gpu}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    actual = [(row["id"], row["seed"]) for row in rows]
    assert len(actual) == len(set(actual)), "Duplicate generation jobs"
    assert set(actual) == expected, f"Missing or unexpected jobs: {expected.symmetric_difference(set(actual))}"
    errors = [row for row in rows if row["status"] != "ok"]
    assert not errors, f"Generation errors: {errors[:2]}"
    for row in rows:
        with Image.open(row["image_path"]) as image:
            assert image.size == (432, 432), row["image_path"]

    out = work / "results"
    out.mkdir(exist_ok=True)
    plain = [row for row in rows if not row["profiled"]]
    stats = {
        "generated": len(rows),
        "prompts": len(tasks),
        "seeds_per_prompt": 4,
        "working_gpu_indices": [0, 1, 2, 3, 4, 6, 7],
        "broken_gpu_index": 5,
        "official_sampling_steps": 50,
        "guidance_scale": 5.0,
        "profiling_excluded_from_latency_medians": True,
        "latency_median_s": median([r["total_s"] for r in plain]),
        "prepare_median_s": median([r["prepare_s"] for r in plain]),
        "denoise_median_s": median([r["denoise_s"] for r in plain]),
        "decode_and_save_median_s": median([r["decode_and_save_s"] for r in plain]),
        "peak_allocated_median_gb": median([r["peak_allocated_gb"] for r in plain]),
        "peak_reserved_median_gb": median([r["peak_reserved_gb"] for r in plain]),
        "by_gpu": {},
        "component_profile_ms": {},
    }
    for gpu in stats["working_gpu_indices"]:
        subset = [r for r in plain if r["gpu_physical"] == gpu]
        stats["by_gpu"][str(gpu)] = {
            "images": sum(r["gpu_physical"] == gpu for r in rows),
            "unprofiled_images": len(subset),
            "median_total_s": median([r["total_s"] for r in subset]),
            "median_denoise_s": median([r["denoise_s"] for r in subset]),
            "max_peak_allocated_gb": max(r["peak_allocated_gb"] for r in rows if r["gpu_physical"] == gpu),
        }
    for category in ("language_backbone", "semantic_transformer", "visual_embedders", "fusion", "flow_head"):
        values = [r["component_gpu_time"][category]["ms"] for r in rows
                  if r.get("component_gpu_time") and category in r["component_gpu_time"]]
        if values:
            stats["component_profile_ms"][category] = {
                "gpu_count": len(values), "median_ms": median(values)
            }
    train_path = work / "logs" / "train_probe_gpu0.json"
    if train_path.exists():
        train = json.loads(train_path.read_text(encoding="utf-8"))
        if train.get("steps") == 200:
            stats["training_probe"] = train
    (out / "stage1_results.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    columns = ["id", "category", "seed", "gpu_physical", "steps", "guidance", "profiled",
               "prepare_s", "denoise_s", "decode_and_save_s", "total_s",
               "peak_allocated_gb", "peak_reserved_gb", "image_path"]
    with (out / "baseline_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["id"], item["seed"])):
            writer.writerow({field: row.get(field) for field in columns})

    thumb = 144
    label_width = 155
    row_height = 170
    canvas = Image.new("RGB", (label_width + 4 * thumb, len(tasks) * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, task in enumerate(tasks):
        y = row_index * row_height
        draw.text((7, y + 10), task["id"], fill="black")
        draw.text((7, y + 30), task["category"], fill="black")
        for col, seed in enumerate(task["seeds"]):
            image_path = work / "images" / f"{task['id']}_seed{seed}.png"
            with Image.open(image_path) as image:
                canvas.paste(image.convert("RGB").resize((thumb, thumb)),
                             (label_width + col * thumb, y))
            draw.text((label_width + col * thumb + 4, y + thumb + 3), str(seed), fill="black")
    canvas.save(out / "contact_sheet.jpg", quality=85)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
