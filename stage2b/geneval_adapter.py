"""Prepare paired Show-o2 tasks and the official GenEval image layout."""

import argparse
import json
from pathlib import Path


def read_metadata(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_tasks(metadata, output: Path, seed_start: int, num_seeds: int,
                per_category: int):
    selected = list(enumerate(metadata))
    if per_category:
        groups = {}
        for index, item in selected:
            groups.setdefault(item["tag"], []).append((index, item))
        selected = []
        for group in groups.values():
            if per_category > len(group):
                raise ValueError("Requested more prompts than a category contains")
            selected.extend(group[round(i * (len(group) - 1) / (per_category - 1))]
                            for i in range(per_category))
        selected.sort()
    tasks = [
        {"id": f"{index:05d}", "category": item["tag"],
         "prompt": item["prompt"],
         "seeds": list(range(seed_start, seed_start + num_seeds))}
        for index, item in selected
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prompts": len(tasks), "seeds": num_seeds, "tasks": str(output)}))


def make_layout(metadata, images: Path, output: Path, variant: str,
                seed_start: int, num_seeds: int, allow_missing: bool):
    missing = []
    ready = 0
    for index, item in enumerate(metadata):
        prompt_id = f"{index:05d}"
        suffix = f"_{variant}" if variant else ""
        sources = [images / f"{prompt_id}_seed{seed}{suffix}.png"
                   for seed in range(seed_start, seed_start + num_seeds)]
        if not all(source.is_file() for source in sources):
            missing.extend(str(source) for source in sources if not source.is_file())
            if allow_missing:
                continue
            continue
        prompt_dir = output / prompt_id
        samples = prompt_dir / "samples"
        samples.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "metadata.jsonl").write_text(
            json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
        for sample_index, source in enumerate(sources):
            destination = samples / f"{sample_index:04d}.png"
            if destination.is_symlink() or destination.exists():
                destination.unlink()
            destination.symlink_to(source.resolve())
        ready += 1
    if missing and not allow_missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first: {missing[:3]}")
    print(json.dumps({"prompts": ready, "images": ready*num_seeds,
                      "missing_images": len(missing), "layout": str(output)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("tasks", "layout"))
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--variant", default="")
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--num-seeds", type=int, default=4)
    parser.add_argument("--per-category", type=int, default=0)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    metadata = read_metadata(args.metadata)
    if args.mode == "tasks":
        write_tasks(metadata, args.output, args.seed_start, args.num_seeds,
                    args.per_category)
    else:
        if args.images is None:
            parser.error("--images is required for layout")
        make_layout(metadata, args.images, args.output, args.variant,
                    args.seed_start, args.num_seeds, args.allow_missing)


if __name__ == "__main__":
    main()
