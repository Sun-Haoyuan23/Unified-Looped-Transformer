"""Freeze prompt-disjoint development/train/validation/sealed-test manifests.

Public API:
  make_splits(metadata, *, excluded_prompts=(), additional_train=(),
      architecture_dev=24, train=120, val=40, test_min=20, seed=20261004,
      legacy_per_category=12, allow_reduced=False) -> dict
  select_smoke(splits, count=4) -> list  # reads only architecture_dev/train.
  write_splits(splits, output_dir, smoke_count=4) -> manifest

CLI: python make_prompt_splits.py --metadata official_metadata.jsonl
     --output-dir splits --exclude old_prompts.json [--exclude another.jsonl]
     [--history-stage1 /remote/work/prompts.json]
     [--tag position] [--train-metadata new_training_prompts.jsonl]
     [--allow-reduced] [--smoke-count 4]

The official old 72-prompt subset is reconstructed from the complete metadata
using the earlier stage2 geneval_adapter.py equally spaced, 12/category rule.
Those prompts (which include the old 12 spatial items) and the stage1 14 prompts
are excluded by default. Additional history should be supplied with --exclude.
Use --legacy-per-category 0 only for a genuinely different input dataset / tests.
Metadata must be complete when reproducing the old subset; filtering by --tag
happens AFTER exclusion. Never pass a prefiltered position-only file pretending
it is the complete official source.

Insufficient data fails unless --allow-reduced explicitly permits a smaller
partition; actual counts and target misses are recorded. External training
metadata can fill the 100-200 train target without consuming validation/test.
All seeds/timepoints from a normalized prompt remain in one partition. The
sealed_test file is separate; its manifest contains only count and content hash.
Smoke selection never reads or samples the sealed-test file or validation set.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata


SPLITS = ("architecture_dev", "train", "val", "sealed_test")


def normalize_prompt(prompt):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a nonempty string")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", prompt)).strip().casefold()


def read_metadata(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if text.lstrip().startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not all(isinstance(row, dict) and "prompt" in row for row in rows):
        raise ValueError(f"{path}: metadata must be JSON array or JSONL objects with prompt")
    return rows


def read_excluded_prompts(path):
    """Read JSON(L) recursive prompt manifests or newline-delimited plain texts."""
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        lines = [line for line in text.splitlines() if line.strip()]
        try:
            data = [json.loads(line) for line in lines]
        except json.JSONDecodeError:
            return [line.strip() for line in lines]
    found = []
    def visit(value):
        if isinstance(value, dict):
            if "prompt" in value and isinstance(value["prompt"], str):
                found.append(value["prompt"])
            for key, item in value.items():
                if key != "prompt":
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    found.append(item)
                else:
                    visit(item)
    visit(data)
    if not found:
        raise ValueError(f"No prompt texts found in exclusion input: {path}")
    return found


def old_geneval_subset(metadata, per_category=12):
    """Exact prompt selection used by the historical stage2 adapter."""
    if per_category < 0:
        raise ValueError("legacy_per_category cannot be negative")
    if per_category == 0:
        return set()
    groups = defaultdict(list)
    for row in metadata:
        groups[row.get("tag", row.get("category", "unknown"))].append(row)
    selected = set()
    for tag, group in groups.items():
        if len(group) < per_category:
            raise ValueError(f"Cannot reconstruct old {per_category}/category subset: {tag} has {len(group)} rows. Supply complete metadata or explicitly set --legacy-per-category 0 for a different dataset.")
        indices = [0] if per_category == 1 else [round(i * (len(group) - 1) / (per_category - 1)) for i in range(per_category)]
        selected.update(normalize_prompt(group[index]["prompt"]) for index in indices)
    return selected


def _canonical_rows(metadata, source):
    seen, result = {}, []
    for index, row in enumerate(metadata):
        normalized = normalize_prompt(row["prompt"])
        tag = row.get("tag", row.get("category", "unknown"))
        if not isinstance(tag, str):
            raise ValueError("tag/category must be a string")
        if normalized in seen:
            if seen[normalized] != tag:
                raise ValueError("Duplicate normalized prompt has conflicting categories")
            continue
        seen[normalized] = tag
        # Preserve full GenEval constraints as metadata, and original index for layout.
        result.append({"prompt_id": f"{index:05d}" if source == "official" else f"train_extra_{index:05d}",
                       "prompt": row["prompt"].strip(), "tag": tag, "source": source,
                       "source_index": index, "metadata": dict(row),
                       "prompt_hash": hashlib.sha256(normalized.encode()).hexdigest()})
    return result


def _balanced_order(rows, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["tag"]].append(row)
    rng = random.Random(seed)
    tags = sorted(groups)
    rng.shuffle(tags)
    for group in groups.values():
        group.sort(key=lambda row: row["prompt_hash"])
        rng.shuffle(group)
    output = []
    while any(groups.values()):
        for tag in tags:
            if groups[tag]:
                output.append(groups[tag].pop())
    return output


def _proportional_counts(targets, budget):
    if budget < len([x for x in targets.values() if x > 0]):
        raise ValueError("Too few unused prompts to allocate nonempty requested partitions")
    counts = {key: int(value > 0) for key, value in targets.items()}
    remaining = budget - sum(counts.values())
    weights = {key: max(value - counts[key], 0) for key, value in targets.items()}
    denominator = sum(weights.values())
    if denominator == 0:
        return counts
    raw = {key: remaining * value / denominator for key, value in weights.items()}
    for key, value in raw.items():
        counts[key] += int(value)
    leftover = budget - sum(counts.values())
    priorities = sorted(targets, key=lambda key: (raw[key] - int(raw[key]), targets[key], key), reverse=True)
    for key in priorities[:leftover]:
        counts[key] += 1
    return counts


def make_splits(metadata, *, excluded_prompts=(), additional_train=(), architecture_dev=24,
                train=120, val=40, test_min=20, seed=20261004, legacy_per_category=12,
                allow_reduced=False, tags=None):
    targets = {"architecture_dev": architecture_dev, "train": train, "val": val}
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in targets.values()) or test_min < 1:
        raise ValueError("Split sizes must be nonnegative integers and test_min at least one")
    if architecture_dev < 1 or val < 1 or train < 1:
        raise ValueError("Development, training and validation must each contain at least one prompt")
    excluded = {normalize_prompt(prompt) for prompt in excluded_prompts}
    legacy = old_geneval_subset(metadata, legacy_per_category)
    excluded.update(legacy)
    official_all = _canonical_rows(metadata, "official")
    official = [row for row in official_all if normalize_prompt(row["prompt"]) not in excluded
                and (tags is None or row["tag"] in tags)]
    extra_all = _canonical_rows(additional_train, "additional_train")
    extra = [row for row in extra_all if normalize_prompt(row["prompt"]) not in excluded]
    official_hashes = {row["prompt_hash"] for row in official_all}
    if any(row["prompt_hash"] in official_hashes for row in extra):
        raise ValueError("Additional training metadata overlaps official metadata; independent source required")
    extra = _balanced_order(extra, seed + 1)
    used_extra = extra[:train]
    official_targets = dict(targets)
    official_targets["train"] -= len(used_extra)
    required = sum(official_targets.values()) + test_min
    shortage = len(official) < required
    actual = dict(official_targets)
    if shortage:
        if not allow_reduced:
            raise ValueError(f"Insufficient unused official prompts: {len(official)} available, {required} needed. Add --train-metadata or explicitly --allow-reduced; never reuse historical/test prompts.")
        if len(official) < 4:
            raise ValueError("At least four unused official prompts are required for isolated partitions")
        reserve = max(1, min(test_min, len(official) // 5))
        actual = _proportional_counts(official_targets, len(official) - reserve)
    ordered = _balanced_order(official, seed)
    splits, offset = {}, 0
    # Assignment is round-robin among available partition quotas so minority
    # categories do not all land in the first slice.
    quotas = dict(actual)
    split_order = ("architecture_dev", "train", "val")
    official_splits = {name: [] for name in split_order}
    while any(quotas.values()):
        for name in split_order:
            if quotas[name] > 0:
                official_splits[name].append(ordered[offset])
                offset += 1
                quotas[name] -= 1
    for name in split_order:
        rows = official_splits[name] + (used_extra if name == "train" else [])
        splits[name] = [dict(row, split=name) for row in rows]
    splits["sealed_test"] = [dict(row, split="sealed_test") for row in ordered[offset:]]
    hashes = [row["prompt_hash"] for name in SPLITS for row in splits[name]]
    if len(hashes) != len(set(hashes)):
        raise AssertionError("Prompt leakage across split boundaries")
    splits["_provenance"] = {"schema_version": 1, "seed": seed, "requested": dict(targets, sealed_test_min=test_min),
                              "actual_counts": {name: len(splits[name]) for name in SPLITS},
                              "reduced_partition": shortage, "historical_prompts_excluded": len(excluded),
                              "legacy_reconstruction_count": len(legacy), "official_source_rows": len(metadata),
                              "extra_training_used": len(used_extra), "tags": None if tags is None else sorted(tags),
                              "prompt_normalization": "Unicode NFKC + whitespace folding + casefold",
                              "selection_uses_model_outcomes": False,
                              "train_target_range_met": 100 <= len(splits["train"]) <= 200,
                              "val_target_range_met": 30 <= len(splits["val"]) <= 50,
                              "architecture_selection_split": "architecture_dev",
                              "checkpoint_selection_split": "val",
                              "test_status": "sealed; unlock only after method/checkpoint freeze"}
    return splits


def select_smoke(splits, count=4):
    if count < 1:
        raise ValueError("Smoke count must be positive")
    # No iteration over SPLITS: sealed_test / val are never inspected here.
    available = list(splits.get("architecture_dev", []))
    if len(available) < count:
        available += list(splits.get("train", []))
    if len(available) < count:
        raise ValueError("Not enough development/train prompts for smoke selection")
    return [dict(row, smoke=True) for row in available[:count]]


def write_splits(splits, output_dir, smoke_count=4):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Reject overwrite rather than silently reassign a previously sealed test.
    files = [output_dir / f"{name}.jsonl" for name in SPLITS]
    files += [output_dir / "smoke.jsonl", output_dir / "manifest.json"]
    if any(path.exists() for path in files):
        raise FileExistsError("Split artifacts already exist; use a new versioned output directory")
    smoke = select_smoke(splits, smoke_count)
    manifest = dict(splits["_provenance"])
    manifest["artifacts"] = {}
    for name in SPLITS:
        payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in splits[name])
        path = output_dir / f"{name}.jsonl"
        path.write_text(payload, encoding="utf-8")
        manifest["artifacts"][name] = {"file": path.name, "prompts": len(splits[name]),
                                      "sha256": hashlib.sha256(payload.encode()).hexdigest()}
        if name != "sealed_test":
            manifest["artifacts"][name]["category_counts"] = dict(Counter(row["tag"] for row in splits[name]))
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in smoke)
    (output_dir / "smoke.jsonl").write_text(payload, encoding="utf-8")
    manifest["smoke"] = {"file": "smoke.jsonl", "prompts": len(smoke),
                         "sources": sorted({row["split"] for row in smoke}),
                         "sealed_test_read": False, "validation_read": False}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude", action="append", type=Path, default=[])
    parser.add_argument("--history-stage1", type=Path,
                        help="Historical stage1 14-prompt manifest; defaults to sibling stage1_showo2/prompts.json")
    parser.add_argument("--train-metadata", action="append", type=Path, default=[])
    parser.add_argument("--tag", action="append")
    parser.add_argument("--architecture-dev", type=int, default=24)
    parser.add_argument("--train", type=int, default=120)
    parser.add_argument("--val", type=int, default=40)
    parser.add_argument("--test-min", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--legacy-per-category", type=int, default=12)
    parser.add_argument("--allow-reduced", action="store_true")
    parser.add_argument("--smoke-count", type=int, default=4)
    args = parser.parse_args(argv)
    excluded = []
    stage1 = args.history_stage1 or (Path(__file__).resolve().parents[1] / "stage1_showo2" / "prompts.json")
    if not stage1.is_file():
        raise FileNotFoundError(f"Historical 14-prompt manifest unavailable: {stage1}; supply --history-stage1 explicitly")
    for path in [stage1] + args.exclude:
        excluded.extend(read_excluded_prompts(path))
    extra = []
    for path in args.train_metadata:
        extra.extend(read_metadata(path))
    splits = make_splits(read_metadata(args.metadata), excluded_prompts=excluded, additional_train=extra,
                         architecture_dev=args.architecture_dev, train=args.train, val=args.val,
                         test_min=args.test_min, seed=args.seed, legacy_per_category=args.legacy_per_category,
                         allow_reduced=args.allow_reduced, tags=None if args.tag is None else set(args.tag))
    manifest = write_splits(splits, args.output_dir, args.smoke_count)
    print(json.dumps({"output_dir": str(args.output_dir), "counts": manifest["actual_counts"],
                      "reduced_partition": manifest["reduced_partition"],
                      "train_target_range_met": manifest["train_target_range_met"],
                      "val_target_range_met": manifest["val_target_range_met"],
                      "smoke_sources": manifest["smoke"]["sources"]}))


if __name__ == "__main__":
    main()
