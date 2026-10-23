"""Compare same-state terminal latent snapshots: actuation, NOT repair quality.

CLI: python diagnose_repair_actuation.py --records experiment --output diag.json
     [--require-hashes] [--reject-all-zero]
API: diagnose(records, *, require_hashes=False, reject_all_zero=False,
              relative_epsilon=1e-12) -> dict

The CLI uses load_action_records, including recursive worker shards. Stored
image_path_sha256/image_sha256 and final_latent_sha256 are verified against file
bytes before Torch loading. Hash mismatch or payload state_id/action mismatch is
a hard error. Old records without hashes stay visibly legacy_unhashed; hashing
their current bytes does NOT retroactively verify provenance. --require-hashes
rejects those old rows. Torch imports lazily and loads snapshots on CPU only.
No generation, optimizer, weight updates or CUDA work are performed.

relative_L2 = ||candidate-baseline||_2 / max(||baseline||_2, epsilon).
Latent exactness includes dtype as well as shape/value equality. Image SHA
equality compares encoded files, not decoded pixels. Per-action aggregates are
descriptive; repeated seeds and fork times are not independent observations.
The report counts unique prompts and (prompt,seed) groups explicitly. A changed
latent does not demonstrate improved geometry, quality, preference or diversity.
An unchanged action is marked all_zero_actuation; known zero controls are expected
to remain exact. Optional --reject-all-zero rejects only other all-zero actions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from score_repair_candidates import load_action_records


ZERO_CONTROL_ACTIONS = {"conditional_zero", "zero", "zero_intervention", "baseline"}
INTERPRETATION = (
    "Terminal latent actuation diagnosis only. Changed latent/image bytes do not prove "
    "semantic repair, improved quality, human preference or generation diversity. "
    "Official geometry/semantics and independent quality evaluation are separate."
)


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_record_files(record, require_hashes=False):
    """Stdlib file verification; also usable without a Torch installation."""
    results = {}
    for field, hash_fields in (("image_path", ("image_path_sha256", "image_sha256")),
                               ("final_latent", ("final_latent_sha256",))):
        value = record.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Missing {field} for {(record.get('state_id'), record.get('action'))}")
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        current = _file_hash(path)
        stored = [record[name] for name in hash_fields if name in record and record[name] is not None]
        if any(not isinstance(value, str) or value.lower() != current for value in stored):
            raise ValueError(f"Stored file hash mismatch for {field} at {(record.get('state_id'), record.get('action'))}")
        if not stored and require_hashes:
            raise ValueError(f"Missing stored hash for {field}; legacy row cannot be provenance-verified")
        results[field] = {"path": str(path), "sha256": current,
                          "provenance_status": "stored_hash_verified" if stored else "legacy_unhashed"}
    return results


def _group_records(records):
    groups = defaultdict(dict)
    for row in records:
        state, action = row.get("state_id"), row.get("action")
        if not isinstance(state, str) or not state or not isinstance(action, str) or not action:
            raise ValueError("Records require nonempty state_id/action")
        if action in groups[state]:
            raise ValueError(f"Duplicate action record: {(state, action)}")
        if not isinstance(row.get("prompt_id"), str) or not row["prompt_id"]:
            raise ValueError("prompt_id required to report independent prompt count")
        groups[state][action] = row
    if not groups:
        raise ValueError("No action records")
    expected = None
    for state, group in groups.items():
        if "baseline" not in group or len(group) < 2:
            raise ValueError(f"Missing baseline or candidate at {state}")
        if expected is None:
            expected = set(group)
        elif expected != set(group):
            raise ValueError(f"Incomplete candidate action set at {state}")
        baseline = group["baseline"]
        for action, row in group.items():
            for field in ("prompt_id", "seed", "flow_index", "true_t", "split", "state_fingerprint"):
                if row.get(field) != baseline.get(field):
                    raise ValueError(f"Same-state {field} mismatch at {(state, action)}")
    return groups


def _load_snapshot(torch, record):
    payload = torch.load(record["final_latent"], map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Terminal snapshot must contain a dictionary payload")
    if payload.get("state_id") != record["state_id"] or payload.get("action") != record["action"]:
        raise ValueError(f"Terminal payload state_id/action mismatch at {(record['state_id'], record['action'])}")
    latent = payload.get("latent")
    if not isinstance(latent, torch.Tensor) or not latent.is_floating_point() or latent.numel() == 0:
        raise ValueError("Terminal latent must be a nonempty floating-point tensor")
    if not bool(torch.isfinite(latent).all()):
        raise ValueError("Terminal latent has nonfinite values")
    return latent.detach().cpu().clone()


def _describe(values):
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def diagnose(records, *, require_hashes=False, reject_all_zero=False, relative_epsilon=1e-12):
    if (isinstance(relative_epsilon, bool) or not isinstance(relative_epsilon, (int, float))
            or not math.isfinite(relative_epsilon) or relative_epsilon <= 0):
        raise ValueError("relative_epsilon must be positive and finite")
    groups = _group_records(records)
    verified = {}
    for state, group in groups.items():
        for action, row in group.items():
            verified[(state, action)] = verify_record_files(row, require_hashes)
    import torch
    pairs = []
    with torch.no_grad():
        for state, group in sorted(groups.items()):
            baseline = _load_snapshot(torch, group["baseline"])
            base_double = baseline.double()
            base_norm = float(torch.linalg.vector_norm(base_double))
            baseline_files = verified[(state, "baseline")]
            for action, row in sorted(group.items()):
                if action == "baseline":
                    continue
                candidate = _load_snapshot(torch, row)
                if candidate.shape != baseline.shape:
                    raise ValueError(f"Terminal latent shape mismatch at {(state, action)}")
                difference = candidate.double() - base_double
                norm = float(torch.linalg.vector_norm(difference))
                exact = candidate.dtype == baseline.dtype and torch.equal(candidate, baseline)
                files = verified[(state, action)]
                hash_verified = all(value["provenance_status"] == "stored_hash_verified"
                                    for value in list(files.values()) + list(baseline_files.values()))
                pairs.append({"state_id": state, "prompt_id": row["prompt_id"], "action": action,
                              "seed": row.get("seed"), "flow_index": row.get("flow_index"),
                              "true_t": row.get("true_t"), "split": row.get("split"),
                              "relative_l2": norm / max(base_norm, relative_epsilon),
                              "mse": float(difference.square().mean()),
                              "max_abs": float(difference.abs().max()), "latent_exact": bool(exact),
                              "baseline_l2": base_norm, "delta_l2": norm,
                              "baseline_dtype": str(baseline.dtype), "candidate_dtype": str(candidate.dtype),
                              "shape": list(baseline.shape),
                              "image_sha_equal": files["image_path"]["sha256"] == baseline_files["image_path"]["sha256"],
                              "provenance_status": "stored_hash_verified" if hash_verified else "legacy_unhashed",
                              "baseline_files": baseline_files, "candidate_files": files})
    # Check again after reading so a changed-in-flight file cannot be reported as
    # the earlier hash's verified snapshot.
    for state, group in groups.items():
        for action, record in group.items():
            for field, info in verified[(state, action)].items():
                if _file_hash(record[field]) != info["sha256"]:
                    raise RuntimeError(f"File changed during diagnosis at {(state, action)}:{field}")
    summaries, zero_actions = {}, []
    for action in sorted({pair["action"] for pair in pairs}):
        subset = [pair for pair in pairs if pair["action"] == action]
        all_zero = all(pair["latent_exact"] for pair in subset)
        expected_zero = action in ZERO_CONTROL_ACTIONS
        if all_zero and not expected_zero:
            zero_actions.append(action)
        summaries[action] = {"n_states": len(subset), "n_prompts": len({p["prompt_id"] for p in subset}),
                             "metrics": {name: _describe([pair[name] for pair in subset])
                                         for name in ("relative_l2", "mse", "max_abs")},
                             "latent_exact_count": sum(pair["latent_exact"] for pair in subset),
                             "image_sha_equal_count": sum(pair["image_sha_equal"] for pair in subset),
                             "all_zero_actuation": all_zero, "expected_zero_control": expected_zero,
                             "status": "expected_zero_control" if all_zero and expected_zero else
                                       "all_zero_actuation" if all_zero else "trajectory_changed",
                             "provenance_counts": dict(Counter(pair["provenance_status"] for pair in subset))}
    if reject_all_zero and zero_actions:
        raise ValueError(f"All-zero actuation in nonzero-intent actions: {zero_actions}")
    prompts = {row["prompt_id"] for group in groups.values() for row in group.values()}
    prompt_seeds = {(row["prompt_id"], row.get("seed")) for group in groups.values() for row in group.values()}
    return {"schema_version": 1, "diagnosis": "terminal latent actuation",
            "interpretation": INTERPRETATION, "device": "cpu", "optimizer_used": False,
            "n_states": len(groups), "n_prompts": len(prompts), "n_prompt_seed_groups": len(prompt_seeds),
            "independence_note": f"{len(groups)} fork states come from {len(prompts)} prompts; repeated seeds/timepoints are not independent samples.",
            "relative_l2_denominator_epsilon": relative_epsilon,
            "all_provenance_hashes_verified": all(pair["provenance_status"] == "stored_hash_verified" for pair in pairs),
            "all_zero_noncontrol_actions": zero_actions, "by_action": summaries, "pairs": pairs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-hashes", action="store_true")
    parser.add_argument("--reject-all-zero", action="store_true")
    args = parser.parse_args(argv)
    result = diagnose(load_action_records(args.records), require_hashes=args.require_hashes,
                      reject_all_zero=args.reject_all_zero)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "states": result["n_states"], "prompts": result["n_prompts"],
                      "all_zero_noncontrol_actions": result["all_zero_noncontrol_actions"],
                      "all_provenance_hashes_verified": result["all_provenance_hashes_verified"]}), flush=True)


if __name__ == "__main__":
    main()
