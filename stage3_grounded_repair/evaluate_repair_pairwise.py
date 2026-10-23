"""Strict same-state paired terminal-outcome evaluation (stdlib only).

Public API: evaluate(records, geneval_rows, score_rows, *, zero_test=None,
    baseline_action='baseline', bootstrap_samples=2000, bootstrap_seed=20261004,
    hps_loss_tolerance=0.0, pick_loss_tolerance=0.0) -> dict.

All three streams must contain exactly one row per (state_id, action). Official
GenEval rows may instead have `filename`; that path must match a unique record's
`image_path`. Scores are {state_id,action,pickscore,hps,siglip}; hpsv2 is accepted
as an alias. Explicit null measurements remain missing, never incorrect/zero.
Missing rows, duplicates, nonfinite measurements and unpaired action sets fail.

Every record must have zero_intervention_test {passed:true,
pixel_max_abs_diff:0, latent_exact:true}, or a global zero_test with those fields
and state_ids covering every evaluated state. Optional cost dict is differenced
field by field. Each state's prompt_id, seed, flow_index and state_fingerprint
must agree across actions when present. Conditional evidence/action cost is not
converted into synthetic FLOPs.

CLI: python evaluate_repair_pairwise.py --records actions.jsonl
     --geneval official_results.jsonl --scores scores.jsonl --output report.json
     [--zero-test zero_test.json] [--paired-output pairs.jsonl]

Oracle maxima are separate, per-metric OFFLINE upper bounds using terminal labels.
They do not establish a deployable controller or an evidence-conditioned loop.
All confidence intervals resample prompts as clusters, retaining all seeds and
timepoints. Exploratory intervals are not adjusted for action/checkpoint selection.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import posixpath
import random
import statistics


METRICS = ("semantic", "pickscore", "hps", "siglip")


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{index}: JSON object required")
                rows.append(row)
    return rows


def _key(row):
    state, action = row.get("state_id"), row.get("action")
    if not isinstance(state, str) or not state or not isinstance(action, str) or not action:
        raise ValueError("Every keyed row needs nonempty string state_id and action")
    return state, action


def _path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("image_path / filename must be a nonempty string")
    # Resolve official layout symlinks when evaluation runs on the image host.
    path = Path(value)
    if path.exists():
        return str(path.resolve()).replace("\\", "/")
    return posixpath.normpath(value.replace("\\", "/"))


def _index(rows, name, path_map=None):
    indexed = {}
    for row in rows:
        if "state_id" in row or "action" in row:
            key = _key(row)
        elif path_map is not None and "filename" in row:
            filename = _path(row["filename"])
            if filename not in path_map:
                raise ValueError(f"Unmapped official GenEval filename: {filename}")
            key = path_map[filename]
        else:
            raise ValueError(f"{name}: key or exact official filename required")
        if key in indexed:
            raise ValueError(f"Duplicate {name} key: {key}")
        indexed[key] = row
    return indexed


def _number(value, field):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number or explicit null")
    return float(value)


def _correct(row):
    if "correct" not in row:
        raise ValueError("Every GenEval row needs correct, including explicit null if missing")
    value = row["correct"]
    if value is None:
        return None
    if isinstance(value, bool) or isinstance(value, int) and value in (0, 1):
        return int(value)
    raise ValueError("GenEval correct must be true/false, 0/1, or null; missing is not incorrect")


def _zero_pass(test):
    return (isinstance(test, dict) and test.get("passed") is True
            and test.get("latent_exact") is True
            and isinstance(test.get("pixel_max_abs_diff"), (int, float))
            and not isinstance(test.get("pixel_max_abs_diff"), bool)
            and test["pixel_max_abs_diff"] == 0)


def _validate_zero(records, zero_test):
    states = {key[0] for key in records}
    globally_verified = set()
    if zero_test is not None:
        if not _zero_pass(zero_test):
            raise ValueError("Matched zero intervention test did not pass exact latent and pixel checks")
        ids = zero_test.get("state_ids")
        if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids) or len(ids) != len(set(ids)):
            raise ValueError("Global zero-test requires unique state_ids coverage")
        globally_verified = set(ids)
        if not states <= globally_verified:
            raise ValueError("Global zero-test does not cover every evaluated state")
    for key, row in records.items():
        if key[0] not in globally_verified and not _zero_pass(row.get("zero_intervention_test")):
            raise ValueError(f"No passed matched zero intervention test for {key}")


def _quality_class(delta, hps_loss_tolerance, pick_loss_tolerance):
    tolerances = {"hps": hps_loss_tolerance, "pickscore": pick_loss_tolerance}
    if any(delta[m] is not None and delta[m] < -tol for m, tol in tolerances.items()):
        return "loss"
    if any(delta[m] is None for m in tolerances):
        return "unknown"
    if any(delta[m] > 0 for m in tolerances):
        return "gain"
    return "preserved"


def classify_pair(base_correct, candidate_correct, delta, hps_loss_tolerance=0, pick_loss_tolerance=0):
    quality = _quality_class(delta, hps_loss_tolerance, pick_loss_tolerance)
    if base_correct is None or candidate_correct is None:
        return "semantic_missing", "missing", quality
    flip = candidate_correct - base_correct
    semantic_status = {1: "wrong_to_correct", -1: "correct_to_wrong"}.get(
        flip, "preserved" if base_correct else "no_repair")
    if flip < 0:
        category = "negative_flip"
    elif flip > 0:
        category = {"loss": "semantic_gain_quality_loss", "unknown": "semantic_gain_quality_unknown"}.get(
            quality, "positive_repair")
    elif quality == "gain":
        category = "quality_only_gain"
    elif quality == "loss":
        category = "quality_only_loss"
    else:
        category = "unchanged"
    return category, semantic_status, quality


def cluster_summary(rows, bootstrap_samples, seed):
    """Mean of observed image/state pairs, with prompt-cluster percentile CI.

    rows: iterable of (prompt_id, baseline_value, candidate_value).
    Missing pairs are omitted visibly. A single prompt yields no inferential CI.
    """
    rows = list(rows)
    complete = [(p, b, c) for p, b, c in rows if b is not None and c is not None]
    if not complete:
        return {"n_total": len(rows), "n_observed_pairs": 0, "n_missing_pairs": len(rows),
                "n_prompts": 0, "baseline_mean": None, "candidate_mean": None,
                "paired_delta": None, "prompt_cluster_bootstrap_95pct_ci": None}
    groups = defaultdict(list)
    for prompt, base, candidate in complete:
        groups[prompt].append(candidate - base)
    prompts = sorted(groups)
    ci = None
    if len(prompts) > 1 and bootstrap_samples > 0:
        rng = random.Random(seed)
        draws = []
        for _ in range(bootstrap_samples):
            sampled = [groups[prompts[rng.randrange(len(prompts))]] for _ in prompts]
            draws.append(sum(sum(group) for group in sampled) / sum(len(group) for group in sampled))
        draws.sort()
        ci = [_quantile(draws, 0.025), _quantile(draws, 0.975)]
    return {"n_total": len(rows), "n_observed_pairs": len(complete),
            "n_missing_pairs": len(rows) - len(complete), "n_prompts": len(prompts),
            "baseline_mean": statistics.mean(b for _, b, _ in complete),
            "candidate_mean": statistics.mean(c for _, _, c in complete),
            "paired_delta": statistics.mean(c - b for _, b, c in complete),
            "prompt_cluster_bootstrap_95pct_ci": ci}


def _quantile(values, q):
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def _metric_seed(seed, label):
    return seed + int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)


def evaluate(records, geneval_rows, score_rows, *, zero_test=None, baseline_action="baseline",
             bootstrap_samples=2000, bootstrap_seed=20261004,
             hps_loss_tolerance=0.0, pick_loss_tolerance=0.0):
    if bootstrap_samples < 0 or hps_loss_tolerance < 0 or pick_loss_tolerance < 0:
        raise ValueError("Bootstrap count and quality-loss tolerances must be nonnegative")
    actions = _index(records, "action records")
    if not actions:
        raise ValueError("No action records")
    _validate_zero(actions, zero_test)
    paths = {}
    for key, row in actions.items():
        if "image_path" in row:
            filename = _path(row["image_path"])
            if filename in paths:
                raise ValueError(f"Ambiguous image_path shared by actions: {filename}")
            paths[filename] = key
    geneval = _index(geneval_rows, "GenEval", paths)
    scores = _index(score_rows, "quality scores")
    if set(geneval) != set(actions) or set(scores) != set(actions):
        raise ValueError("Action records, official GenEval and quality scores must have identical key sets")
    by_state = defaultdict(dict)
    measurements = {}
    for key, row in actions.items():
        if not isinstance(row.get("prompt_id"), str) or not row["prompt_id"]:
            raise ValueError(f"prompt_id is required for prompt-cluster bootstrap: {key}")
        by_state[key[0]][key[1]] = row
        quality = scores[key]
        if "hps" in quality and "hpsv2" in quality and quality["hps"] != quality["hpsv2"]:
            raise ValueError("Conflicting hps and hpsv2 scores")
        for field in ("pickscore", "siglip"):
            if field not in quality:
                raise ValueError(f"{field} must be present, using explicit null if unmeasured")
        if "hps" not in quality and "hpsv2" not in quality:
            raise ValueError("hps or hpsv2 must be present, using null if unmeasured")
        measurements[key] = {"semantic": _correct(geneval[key]),
                             "pickscore": _number(quality["pickscore"], "pickscore"),
                             "hps": _number(quality.get("hps", quality.get("hpsv2")), "hps"),
                             "siglip": _number(quality["siglip"], "siglip")}
        cost = row.get("cost", {})
        if not isinstance(cost, dict):
            raise ValueError("cost must be a dictionary")
        for field, value in cost.items():
            parsed = _number(value, f"cost.{field}")
            if parsed is not None and parsed < 0:
                raise ValueError("Action cost cannot be negative")
    expected_actions = None
    for state, group in by_state.items():
        if baseline_action not in group or len(group) < 2:
            raise ValueError(f"State {state} needs baseline and at least one candidate")
        if expected_actions is None:
            expected_actions = set(group)
        elif set(group) != expected_actions:
            raise ValueError(f"Incomplete candidate action set at {state}")
        for field in ("prompt_id", "seed", "flow_index", "true_t", "split", "state_fingerprint"):
            values = [row.get(field) for row in group.values()]
            if any(value != values[0] for value in values):
                raise ValueError(f"Same-state pairing mismatch for {field} at {state}")
        if any(row.get("state_fingerprint") is not None for row in group.values()):
            if not isinstance(group[baseline_action]["state_fingerprint"], str) or not group[baseline_action]["state_fingerprint"]:
                raise ValueError("state_fingerprint must be a nonempty string")
        official_tags = [geneval[(state, action)].get("tag") for action in group]
        if any(tag != official_tags[0] for tag in official_tags):
            raise ValueError(f"Official GenEval category mismatch at {state}")
    pairs = []
    candidate_actions = sorted(expected_actions - {baseline_action})
    for state in sorted(by_state):
        base = measurements[(state, baseline_action)]
        base_record = by_state[state][baseline_action]
        for action in candidate_actions:
            candidate = measurements[(state, action)]
            record = by_state[state][action]
            delta = {m: None if base[m] is None or candidate[m] is None else candidate[m] - base[m]
                     for m in METRICS}
            category, semantic_status, quality = classify_pair(
                base["semantic"], candidate["semantic"], delta, hps_loss_tolerance, pick_loss_tolerance)
            cost_keys = set(base_record.get("cost", {})) | set(record.get("cost", {}))
            costs = {name: None if base_record.get("cost", {}).get(name) is None or record.get("cost", {}).get(name) is None
                     else record["cost"][name] - base_record["cost"][name] for name in sorted(cost_keys)}
            pairs.append({"state_id": state, "prompt_id": record["prompt_id"], "action": action,
                          "seed": record.get("seed"), "flow_index": record.get("flow_index"),
                          "true_t": record.get("true_t"), "split": record.get("split"),
                          "tag": geneval[(state, action)].get("tag", record.get("tag", "unknown")),
                          "baseline_correct": base["semantic"], "candidate_correct": candidate["semantic"],
                          "semantic_flip": delta["semantic"], "semantic_status": semantic_status,
                          "classification": category, "quality_guard": quality,
                          "pick_delta": delta["pickscore"], "hps_delta": delta["hps"],
                          "siglip_delta": delta["siglip"], "compute_delta": costs,
                          "baseline_cost": base_record.get("cost", {}), "candidate_cost": record.get("cost", {}),
                          "baseline_metrics": base, "candidate_metrics": candidate})
    summaries = {}
    for action in candidate_actions:
        action_pairs = [pair for pair in pairs if pair["action"] == action]
        summaries[action] = _summarize_pairs(action_pairs, bootstrap_samples, bootstrap_seed)
        categories = sorted({p["tag"] or "unknown" for p in action_pairs})
        summaries[action]["by_category"] = {
            tag: _summarize_pairs([p for p in action_pairs if (p["tag"] or "unknown") == tag],
                                  bootstrap_samples, bootstrap_seed)
            for tag in categories}
    oracle = {}
    for metric in METRICS:
        oracle_rows, selected = [], Counter()
        for state, group in sorted(by_state.items()):
            base = measurements[(state, baseline_action)][metric]
            values = {action: measurements[(state, action)][metric] for action in group}
            if base is None or any(value is None for value in values.values()):
                oracle_rows.append((group[baseline_action]["prompt_id"], None, None))
                continue
            best = max(values, key=lambda a: (values[a], a == baseline_action, a))
            selected[best] += 1
            oracle_rows.append((group[baseline_action]["prompt_id"], base, values[best]))
        oracle[metric] = cluster_summary(oracle_rows, bootstrap_samples,
                                        _metric_seed(bootstrap_seed, "oracle:" + metric))
        oracle[metric]["selected_action_counts"] = dict(selected)
    quality_oracle_rows, quality_choices = [], Counter()
    for state, group in sorted(by_state.items()):
        base = measurements[(state, baseline_action)]
        if any(base[m] is None for m in ("semantic", "hps", "pickscore")):
            quality_oracle_rows.append((group[baseline_action]["prompt_id"], None, None))
            continue
        eligible = {baseline_action: base["semantic"]}
        complete = True
        for action in candidate_actions:
            value = measurements[(state, action)]
            if any(value[m] is None for m in ("semantic", "hps", "pickscore")):
                complete = False
                break
            if value["hps"] - base["hps"] >= -hps_loss_tolerance and value["pickscore"] - base["pickscore"] >= -pick_loss_tolerance:
                eligible[action] = value["semantic"]
        if not complete:
            quality_oracle_rows.append((group[baseline_action]["prompt_id"], None, None))
            continue
        best = max(eligible, key=lambda a: (eligible[a], a == baseline_action, a))
        quality_choices[best] += 1
        quality_oracle_rows.append((group[baseline_action]["prompt_id"], base["semantic"], eligible[best]))
    guarded_oracle = cluster_summary(quality_oracle_rows, bootstrap_samples,
                                     _metric_seed(bootstrap_seed, "guarded_oracle"))
    guarded_oracle["selected_action_counts"] = dict(quality_choices)
    return {"schema_version": 1, "verification_status": "ANALYZED",
            "evaluation_unit": "same-state paired terminal continuation",
            "n_states": len(by_state), "n_prompts": len({r["prompt_id"] for r in actions.values()}),
            "matched_zero_intervention_passed": True,
            "candidate_actions": candidate_actions, "quality_loss_tolerances": {
                "hps": hps_loss_tolerance, "pickscore": pick_loss_tolerance},
            "bootstrap": {"cluster": "prompt_id", "samples": bootstrap_samples, "seed": bootstrap_seed,
                          "estimate": "observed-state-pair mean", "multiple_selection_adjusted": False},
            "by_action": summaries,
            "offline_oracle": {"label": "terminal-label offline upper bound; not deployable or independent evidence",
                               "metrics_choose_different_actions": True, "per_metric": oracle,
                               "quality_guarded_semantic": guarded_oracle},
            "controller_gate": {"automatically_authorized": False,
                                "required": "Independent fixed-repair validation: repeatable semantic repairs, preservation, no systematic HPS/Pick decline; oracle alone is insufficient."},
            "pairs": pairs}


def _summarize_pairs(pairs, bootstrap_samples, seed):
    flips = [p for p in pairs if p["semantic_flip"] is not None]
    wrong = [p for p in flips if p["baseline_correct"] == 0]
    correct = [p for p in flips if p["baseline_correct"] == 1]
    return {"n_pairs": len(pairs), "classification_counts": dict(Counter(p["classification"] for p in pairs)),
            "semantic_transition_counts": dict(Counter(p["semantic_status"] for p in pairs)),
            "wrong_to_correct": sum(p["semantic_flip"] == 1 for p in flips),
            "correct_to_wrong": sum(p["semantic_flip"] == -1 for p in flips),
            "baseline_wrong_observed": len(wrong), "baseline_correct_observed": len(correct),
            "repair_rate_among_baseline_wrong": None if not wrong else sum(p["semantic_flip"] == 1 for p in wrong) / len(wrong),
            "regression_rate_among_baseline_correct": None if not correct else sum(p["semantic_flip"] == -1 for p in correct) / len(correct),
            "metrics": {metric: cluster_summary(
                [(p["prompt_id"], p["baseline_metrics"][metric], p["candidate_metrics"][metric]) for p in pairs],
                bootstrap_samples, _metric_seed(seed, metric)) for metric in METRICS},
            "cost_metrics": {metric: cluster_summary(
                [(p["prompt_id"], p["baseline_cost"].get(metric), p["candidate_cost"].get(metric)) for p in pairs],
                bootstrap_samples, _metric_seed(seed, "cost:" + metric))
                for metric in sorted({m for p in pairs for m in p["compute_delta"]})}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--geneval", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--zero-test", type=Path)
    parser.add_argument("--baseline-action", default="baseline")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paired-output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261004)
    parser.add_argument("--hps-loss-tolerance", type=float, default=0.0)
    parser.add_argument("--pick-loss-tolerance", type=float, default=0.0)
    args = parser.parse_args(argv)
    result = evaluate(read_jsonl(args.records), read_jsonl(args.geneval), read_jsonl(args.scores),
                      zero_test=None if args.zero_test is None else json.loads(args.zero_test.read_text(encoding="utf-8-sig")),
                      baseline_action=args.baseline_action, bootstrap_samples=args.bootstrap_samples,
                      bootstrap_seed=args.bootstrap_seed, hps_loss_tolerance=args.hps_loss_tolerance,
                      pick_loss_tolerance=args.pick_loss_tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.paired_output:
        args.paired_output.parent.mkdir(parents=True, exist_ok=True)
        args.paired_output.write_text("".join(json.dumps(p, ensure_ascii=False, allow_nan=False) + "\n" for p in result["pairs"]), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "states": result["n_states"], "prompts": result["n_prompts"],
                      "actions": result["candidate_actions"], "zero_intervention": "passed"}))


if __name__ == "__main__":
    main()
