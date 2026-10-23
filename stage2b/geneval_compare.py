"""Paired GenEval comparison with per-category deltas and uncertainty."""

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def read_results(path: Path):
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        image_path = Path(row["filename"])
        key = (image_path.parent.parent.name, image_path.stem)
        if key in records:
            raise ValueError(f"Duplicate GenEval item: {key}")
        records[key] = {"tag": row["tag"], "correct": int(row["correct"])}
    return records


def summarize(keys, base, candidate, seed=20261003):
    base_scores = [base[key]["correct"] for key in keys]
    candidate_scores = [candidate[key]["correct"] for key in keys]
    deltas = [b - a for a, b in zip(base_scores, candidate_scores)]
    rng = random.Random(seed)
    n = len(keys)
    bootstraps = sorted(
        statistics.mean(deltas[rng.randrange(n)] for _ in range(n))
        for _ in range(10000)
    )
    return {"n_images": n, "baseline": statistics.mean(base_scores),
            "candidate": statistics.mean(candidate_scores),
            "paired_delta": statistics.mean(deltas),
            "bootstrap_95pct_ci": [bootstraps[250], bootstraps[9749]],
            "candidate_only_correct": sum(x == 1 for x in deltas),
            "baseline_only_correct": sum(x == -1 for x in deltas)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = read_results(args.baseline)
    candidate = read_results(args.candidate)
    if set(base) != set(candidate):
        raise ValueError(f"Unpaired results: {len(base)} baseline and "
                         f"{len(candidate)} candidate images")
    groups = defaultdict(list)
    for key in sorted(base):
        if base[key]["tag"] != candidate[key]["tag"]:
            raise ValueError(f"Category mismatch at {key}")
        groups[base[key]["tag"]].append(key)
    output = {"overall": summarize(sorted(base), base, candidate),
              "by_category": {tag: summarize(keys, base, candidate)
                              for tag, keys in sorted(groups.items())}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
