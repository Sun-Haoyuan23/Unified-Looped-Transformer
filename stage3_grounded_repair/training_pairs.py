"""Stdlib-only gate for B1 terminal-outcome teachers and preservation states.

API:
  selected_pairs(report, records) -> {state_id: copied_teacher_action_record}
  preservation_states(report, records) -> sorted list[str]
  validate_training_pairs(report, records) -> descriptive gate summary

Each API validates BOTH cohorts and raises TrainingGateError unless there is at
least one TRAIN wrong-to-correct teacher with nonmissing nonnegative HPS/Pick
deltas AND at least one genuine baseline-correct TRAIN preservation state.
This is a data-readiness gate, not a claim that a trained repairer is effective.
Validation/development states may be present in the report but are never selected.

Canonical paired_report fields are flat pick_delta/hps_delta/siglip_delta, plus
baseline_metrics/candidate_metrics. Missing row/field, duplicate key, nonfinite
score, inconsistent delta, split or same-state fingerprint fail. Explicit null
measurements remain missing and cannot qualify a teacher. Records must cover the
report's entire (state_id, action) candidate set plus their baselines exactly.
The current report schema carries fingerprints in action records; those baseline
and candidate fingerprints are required to agree. If a report pair includes an
additional fingerprint, it must agree too. This module cannot verify latent file
bytes: the trainer must also validate loaded target payload state_id/action and
captured latent hash before using it.

A prompt_repair/opposite_prompt or layout_plan/opposite_layout teacher may supply an independently verified
successful continuation target. Its teacher_note explicitly identifies an offline
prompt-edit interface, not evidence that a learned shared-loop repairer works.
No controller permission is granted by this gate.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
import unicodedata


class TrainingGateError(ValueError):
    """The training data do not satisfy the same-state B1 readiness contract."""


QUALITY_FIELDS = {"pickscore": "pick_delta", "hps": "hps_delta", "siglip": "siglip_delta"}
PAIR_STATE_FIELDS = ("prompt_id", "seed", "flow_index", "true_t", "split")
ALLOWED_SPLITS = {"train", "val", "architecture_dev", "sealed_test"}


def _key(row, label):
    if not isinstance(row, dict):
        raise TrainingGateError(f"{label} must contain dictionary rows")
    state, action = row.get("state_id"), row.get("action")
    if not isinstance(state, str) or not state or not isinstance(action, str) or not action:
        raise TrainingGateError(f"{label} row requires nonempty state_id/action")
    return state, action


def _index(rows, label):
    indexed = {}
    for row in rows:
        item = _key(row, label)
        if item in indexed:
            raise TrainingGateError(f"Duplicate {label} key: {item}")
        indexed[item] = row
    return indexed


def _number(value, field):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TrainingGateError(f"Nonfinite or nonnumeric quality measurement: {field}")
    return float(value)


def _correct(value, field):
    if value is None:
        return None
    if isinstance(value, bool) or isinstance(value, int) and value in (0, 1):
        return int(value)
    raise TrainingGateError(f"{field} must be 0/1, bool or explicit null")


def _require(row, field, label):
    if field not in row:
        raise TrainingGateError(f"Missing {field} in {label}")
    return row[field]


def _normalized_prompt(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def _validate_record(row, label):
    fingerprint = _require(row, "state_fingerprint", label)
    if not isinstance(fingerprint, str) or not fingerprint:
        raise TrainingGateError(f"Missing or invalid state_fingerprint in {label}")
    for field in PAIR_STATE_FIELDS:
        _require(row, field, label)
    if not isinstance(row["prompt_id"], str) or not row["prompt_id"]:
        raise TrainingGateError(f"Invalid prompt_id in {label}")
    if not isinstance(row["split"], str) or row["split"] not in ALLOWED_SPLITS:
        raise TrainingGateError(f"Unrecognized split in {label}: {row['split']}")
    for field in ("seed", "flow_index"):
        if isinstance(row[field], bool) or not isinstance(row[field], int) or row[field] < 0:
            raise TrainingGateError(f"Invalid {field} in {label}")
    true_t = _number(row["true_t"], "true_t")
    if true_t is None or not 0 <= true_t <= 1:
        raise TrainingGateError(f"Invalid true_t in {label}")
    zero = _require(row, "zero_intervention_test", label)
    if (not isinstance(zero, dict) or zero.get("passed") is not True
            or zero.get("latent_exact") is not True
            or isinstance(zero.get("pixel_max_abs_diff"), bool)
            or zero.get("pixel_max_abs_diff") != 0):
        raise TrainingGateError(f"Matched zero intervention failed or missing in {label}")


def _validated(report, records):
    if not isinstance(report, dict) or report.get("matched_zero_intervention_passed") is not True:
        raise TrainingGateError("NO-GO: matched zero intervention gate has not passed")
    pair_rows = report.get("pairs")
    if not isinstance(pair_rows, list) or not pair_rows:
        raise TrainingGateError("Missing or empty canonical paired_report.pairs")
    record_index = _index(records, "action record")
    pairs = _index(pair_rows, "report pair")
    if not record_index:
        raise TrainingGateError("Missing action records")
    expected = {item for item in record_index if item[1] != "baseline"}
    if set(pairs) != expected:
        raise TrainingGateError("Missing or extra report pairs; exact record/report alignment required")
    states = defaultdict(dict)
    prompt_splits = defaultdict(set)
    text_splits = defaultdict(set)
    for item, row in record_index.items():
        _validate_record(row, str(item))
        states[item[0]][item[1]] = row
        prompt_splits[row["prompt_id"]].add(row["split"])
        if "prompt" in row:
            if not isinstance(row["prompt"], str) or not row["prompt"].strip():
                raise TrainingGateError(f"Invalid original prompt at {item}")
            text_splits[_normalized_prompt(row["prompt"])].add(row["split"])
    if any(len(splits) > 1 for splits in prompt_splits.values()) or any(len(splits) > 1 for splits in text_splits.values()):
        raise TrainingGateError("Prompt split leakage across train/validation/development/test")
    action_set = None
    for state, group in states.items():
        if "baseline" not in group or len(group) < 2:
            raise TrainingGateError(f"Missing baseline or candidate for {state}")
        if action_set is None:
            action_set = set(group)
        elif set(group) != action_set:
            raise TrainingGateError(f"Incomplete candidate action set at {state}")
        baseline = group["baseline"]
        for action, row in group.items():
            for field in PAIR_STATE_FIELDS + ("state_fingerprint",):
                if row[field] != baseline[field]:
                    raise TrainingGateError(f"Paired {field} mismatch at {(state, action)}")
            if "prompt" in baseline and row.get("prompt") != baseline["prompt"]:
                raise TrainingGateError(f"Original prompt mismatch at {(state, action)}")
    if "n_states" in report and report["n_states"] != len(states):
        raise TrainingGateError("Report n_states disagrees with paired records")
    if "candidate_actions" in report and set(report["candidate_actions"]) != action_set - {"baseline"}:
        raise TrainingGateError("Report candidate_actions disagrees with paired records")
    baseline_labels, baseline_metrics = {}, {}
    teacher_ranks, correct_states = {}, set()
    skipped_quality_missing, skipped_quality_loss = 0, 0
    for item, pair in sorted(pairs.items()):
        state, action = item
        record = record_index[item]
        for field in PAIR_STATE_FIELDS:
            if _require(pair, field, str(item)) != record[field]:
                raise TrainingGateError(f"Report/action {field} mismatch at {item}")
        for field in ("state_fingerprint", "baseline_state_fingerprint", "candidate_state_fingerprint"):
            if field in pair and pair[field] != record["state_fingerprint"]:
                raise TrainingGateError(f"Report/action fingerprint mismatch at {item}")
        base_correct = _correct(_require(pair, "baseline_correct", str(item)), "baseline_correct")
        candidate_correct = _correct(_require(pair, "candidate_correct", str(item)), "candidate_correct")
        if state in baseline_labels and baseline_labels[state] != base_correct:
            raise TrainingGateError(f"Conflicting baseline semantic labels for {state}")
        baseline_labels[state] = base_correct
        base = _require(pair, "baseline_metrics", str(item))
        candidate = _require(pair, "candidate_metrics", str(item))
        if not isinstance(base, dict) or not isinstance(candidate, dict):
            raise TrainingGateError("baseline_metrics/candidate_metrics must be dictionaries")
        if _correct(_require(base, "semantic", str(item)), "baseline_metrics.semantic") != base_correct:
            raise TrainingGateError(f"Baseline semantic metric mismatch at {item}")
        if _correct(_require(candidate, "semantic", str(item)), "candidate_metrics.semantic") != candidate_correct:
            raise TrainingGateError(f"Candidate semantic metric mismatch at {item}")
        delta = {}
        current_base = {}
        for metric, flat_field in QUALITY_FIELDS.items():
            b = _number(_require(base, metric, str(item)), "baseline." + metric)
            c = _number(_require(candidate, metric, str(item)), "candidate." + metric)
            measured_delta = _number(_require(pair, flat_field, str(item)), flat_field)
            expected_delta = None if b is None or c is None else c - b
            if (measured_delta is None) != (expected_delta is None):
                raise TrainingGateError(f"Missing quality/delta inconsistency at {item}:{metric}")
            if expected_delta is not None and not math.isclose(measured_delta, expected_delta, rel_tol=1e-9, abs_tol=1e-12):
                raise TrainingGateError(f"Quality delta arithmetic mismatch at {item}:{metric}")
            delta[metric] = measured_delta
            current_base[metric] = b
        if state in baseline_metrics and current_base != baseline_metrics[state]:
            raise TrainingGateError(f"Conflicting baseline quality metrics for {state}")
        baseline_metrics[state] = current_base
        if record["split"] != "train":
            continue
        if base_correct == 1:
            correct_states.add(state)
        if base_correct != 0 or candidate_correct != 1:
            continue
        if any(delta[m] is None for m in ("pickscore", "hps")):
            skipped_quality_missing += 1
            continue
        if any(delta[m] < 0 for m in ("pickscore", "hps")):
            skipped_quality_loss += 1
            continue
        target = record.get("final_latent")
        if not isinstance(target, str) or not target:
            raise TrainingGateError(f"Selected terminal teacher has no final_latent path at {item}")
        rank = delta["hps"], delta["pickscore"], action
        if state not in teacher_ranks or rank > teacher_ranks[state][0]:
            teacher_ranks[state] = rank, record
    if not teacher_ranks:
        raise TrainingGateError("NO-GO: no train wrong-to-correct teacher preserving both HPS and PickScore")
    if not correct_states:
        raise TrainingGateError("NO-GO: no actual baseline-correct train preservation state")
    selected = {}
    for state, (_, record) in sorted(teacher_ranks.items()):
        row = dict(record)
        if row["action"] in ("prompt_repair", "opposite_prompt", "layout_plan", "opposite_layout"):
            note = "Offline prompt-edit teacher interface; positive teacher outcome is not evidence of a learned shared-loop repairer."
        elif row["action"] in ("regional_flow", "opposite_regional"):
            note = "Offline regional velocity teacher interface using additional full-network calls; positive outcome is not evidence of shared-loop repair efficacy. Score against the original prompt."
        elif row["action"] == "extra_ode":
            note = "Extra ODE teacher interface; positive outcome is not evidence of shared-loop recurrence efficacy."
        elif "probe" in row["action"]:
            note = "Untrained deterministic evidence probe teacher; evaluate the trained repairer independently."
        else:
            note = "Same-state terminal teacher selected by official semantics and both independent quality guards."
        row["training_teacher_note"] = note
        selected[state] = row
    return selected, sorted(correct_states), {
        "skipped_train_semantic_repairs_with_missing_quality": skipped_quality_missing,
        "skipped_train_semantic_repairs_with_quality_loss": skipped_quality_loss,
    }


def selected_pairs(report, records):
    """Return one verified train teacher per state; also require preservation cohort."""
    selected, _, _ = _validated(report, records)
    return selected


def preservation_states(report, records):
    """Return genuine baseline-correct train state IDs; also require positive teachers."""
    _, correct, _ = _validated(report, records)
    return correct


def validate_training_pairs(report, records):
    """Describe readiness without granting controller/validation or scientific claims."""
    selected, correct, diagnostics = _validated(report, records)
    prompts = {row["prompt_id"] for row in selected.values()}
    return {"data_ready": True, "selected_teacher_states": len(selected),
            "selected_teacher_prompts": len(prompts), "preservation_states": len(correct),
            "teacher_action_counts": dict(Counter(row["action"] for row in selected.values())),
            "teacher_actions": {state: row["action"] for state, row in selected.items()},
            "allowed_split": "train", "controller_authorized": False,
            "independent_terminal_validation_required": True,
            "semantic_guard": "official baseline wrong -> candidate correct",
            "quality_guard": "both finite nonmissing HPS/Pick deltas >= 0",
            **diagnostics}
