"""Deterministic, category-balanced PartiPrompts pilot split.

The original 14 stage-1 prompts remain a separate unseen evaluation set.
PartiPrompts is from https://github.com/google-research/parti .
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def digest(value):
    return hashlib.sha256(("stage2b-20261003:" + value).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-category", type=int, default=10)
    parser.add_argument("--val-per-category", type=int, default=2)
    args = parser.parse_args()
    lines = args.input.read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t") == ["Prompt", "Category", "Challenge", "Note"]
    categories = defaultdict(list)
    seen = set()
    for line in lines[1:]:
        prompt, category, challenge, note = line.split("\t", 3)
        prompt = prompt.strip()
        if not prompt or prompt.casefold() in seen:
            continue
        seen.add(prompt.casefold())
        categories[category].append((prompt, challenge, note))
    manifest = {"source": "google-research/parti/PartiPrompts.tsv",
                "selection": "SHA256 deterministic, category-balanced",
                "train": [], "val": []}
    for category in sorted(categories):
        entries = sorted(categories[category], key=lambda x: digest(x[0]))
        needed = args.train_per_category + args.val_per_category
        if len(entries) < needed:
            raise RuntimeError(f"Not enough prompts in {category}: {len(entries)}")
        selected = entries[:needed]
        for index, (prompt, challenge, note) in enumerate(selected):
            split = "train" if index < args.train_per_category else "val"
            manifest[split].append({"id": f"p{len(manifest[split]):04d}",
                                    "prompt": prompt, "category": category,
                                    "challenge": challenge, "seed": 20261003 +
                                    len(manifest[split]) * 17})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"train": len(manifest["train"]), "val": len(manifest["val"]),
                      "categories": len(categories)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
