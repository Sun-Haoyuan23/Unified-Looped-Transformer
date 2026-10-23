"""Add category-balanced PartiPrompts training prompts without touching pilot val."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from make_manifest import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-train-per-category", type=int, default=35)
    args = parser.parse_args()
    pilot = json.loads(args.pilot.read_text(encoding="utf-8"))
    existing = {job["prompt"].casefold() for job in pilot["train"] + pilot["val"]}
    categories = defaultdict(list)
    for line in args.input.read_text(encoding="utf-8").splitlines()[1:]:
        prompt, category, challenge, note = line.split("\t", 3)
        prompt = prompt.strip()
        if prompt:
            categories[category].append((prompt, challenge))
    result = {**pilot, "selection": pilot["selection"] +
              "; extended train excludes all pilot validation prompts",
              "train": list(pilot["train"]), "val": list(pilot["val"])}
    additions = []
    for category in sorted(categories):
        have = sum(job["category"] == category for job in result["train"])
        for prompt, challenge in sorted(categories[category], key=lambda x: digest(x[0])):
            if have >= args.total_train_per_category:
                break
            if prompt.casefold() in existing:
                continue
            existing.add(prompt.casefold())
            item = {"id": f"e{len(additions):04d}", "prompt": prompt,
                    "category": category, "challenge": challenge,
                    "seed": 20261203 + len(additions) * 17}
            additions.append(item)
            have += 1
        if have < args.total_train_per_category:
            raise RuntimeError(f"Insufficient prompts in {category}: {have}")
    result["train"].extend(additions)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"train": len(result["train"]), "new": len(additions),
                      "val": len(result["val"])}))


if __name__ == "__main__":
    main()
