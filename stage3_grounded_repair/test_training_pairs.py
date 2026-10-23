"""Hermetic B1 data gate tests using actual flat paired-report schema."""

from copy import deepcopy
import unittest

from evaluate_repair_pairwise import evaluate
from training_pairs import (TrainingGateError, preservation_states, selected_pairs,
                            validate_training_pairs)


def fixture():
    records, semantics, scores = [], [], []
    zero = {"passed": True, "latent_exact": True, "pixel_max_abs_diff": 0}
    for state, split, base_correct in (("repair", "train", 0), ("preserve", "train", 1),
                                       ("dev", "architecture_dev", 0), ("validation", "val", 0)):
        for action in ("baseline", "conditional_revisit", "prompt_repair"):
            item = {"state_id": state, "action": action}
            records.append({**item, "prompt_id": "prompt_" + state, "prompt": "Original prompt " + state,
                            "seed": 101, "flow_index": 30, "true_t": .345, "split": split,
                            "state_fingerprint": "captured-" + state,
                            "zero_intervention_test": dict(zero),
                            "image_path": f"/synthetic/{state}/{action}/samples/0000.png",
                            "final_latent": f"/synthetic/{state}/{action}/final_latent.pt"})
            correct = base_correct if action == "baseline" else 1
            semantics.append({**item, "correct": correct, "tag": "position"})
            gain = {"baseline": 0., "conditional_revisit": .01, "prompt_repair": .02}[action]
            scores.append({**item, "pickscore": 22. + gain, "hps": .29 + gain, "siglip": .2})
    report = evaluate(records, semantics, scores, bootstrap_samples=0)
    return report, records


def pair(report, state, action):
    return next(row for row in report["pairs"] if row["state_id"] == state and row["action"] == action)


def set_quality(report, state, action, metric, value):
    row = pair(report, state, action)
    row["candidate_metrics"][metric] = value
    flat = {"hps": "hps_delta", "pickscore": "pick_delta", "siglip": "siglip_delta"}[metric]
    row[flat] = None if value is None else value - row["baseline_metrics"][metric]


class TrainingPairsTests(unittest.TestCase):
    def test_flat_schema_success_and_preservation(self):
        report, records = fixture()
        teachers = selected_pairs(report, records)
        self.assertEqual(set(teachers), {"repair"})
        self.assertEqual(teachers["repair"]["action"], "prompt_repair")
        self.assertIn("Offline prompt-edit teacher", teachers["repair"]["training_teacher_note"])
        self.assertEqual(preservation_states(report, records), ["preserve"])
        summary = validate_training_pairs(report, records)
        self.assertEqual(summary["teacher_action_counts"], {"prompt_repair": 1})
        self.assertFalse(summary["controller_authorized"])
        self.assertTrue(summary["independent_terminal_validation_required"])
        self.assertNotIn("delta", pair(report, "repair", "prompt_repair"))
        # Public API returns a copy and cannot mutate the input action record.
        self.assertNotIn("training_teacher_note", records[2])

    def test_quality_loss_not_selected_even_with_semantic_gain(self):
        report, records = fixture()
        set_quality(report, "repair", "prompt_repair", "hps", .28)
        teachers = selected_pairs(report, records)
        self.assertEqual(teachers["repair"]["action"], "conditional_revisit")
        for metric in ("hps", "pickscore"):
            report, records = fixture()
            for action in ("prompt_repair", "conditional_revisit"):
                set_quality(report, "repair", action, metric, .1)
            with self.assertRaisesRegex(TrainingGateError, "no train wrong-to-correct"):
                selected_pairs(report, records)

    def test_explicit_missing_quality_not_zero_or_positive(self):
        report, records = fixture()
        set_quality(report, "repair", "prompt_repair", "hps", None)
        self.assertEqual(selected_pairs(report, records)["repair"]["action"], "conditional_revisit")
        for action in ("conditional_revisit", "prompt_repair"):
            set_quality(report, "repair", action, "hps", None)
        with self.assertRaisesRegex(TrainingGateError, "no train wrong-to-correct"):
            selected_pairs(report, records)

    def test_dev_and_validation_do_not_supply_training_positive(self):
        report, records = fixture()
        for action in ("conditional_revisit", "prompt_repair"):
            row = pair(report, "repair", action)
            row["candidate_correct"] = 0
            row["candidate_metrics"]["semantic"] = 0
        # Development and validation still contain semantic repairs, but no train
        # positive remains. Both APIs must refuse readiness.
        for function in (selected_pairs, preservation_states):
            with self.assertRaisesRegex(TrainingGateError, "no train wrong-to-correct"):
                function(report, records)

    def test_need_real_correct_train_preservation(self):
        report, records = fixture()
        for row in report["pairs"]:
            if row["state_id"] == "preserve":
                row["baseline_correct"] = 0
                row["baseline_metrics"]["semantic"] = 0
        with self.assertRaisesRegex(TrainingGateError, "baseline-correct train preservation"):
            selected_pairs(report, records)

    def test_duplicate_record_pair_and_missing_alignment_rejected(self):
        report, records = fixture()
        with self.assertRaisesRegex(TrainingGateError, "Duplicate action record"):
            selected_pairs(report, records + [deepcopy(records[0])])
        report["pairs"].append(deepcopy(report["pairs"][0]))
        with self.assertRaisesRegex(TrainingGateError, "Duplicate report pair"):
            selected_pairs(report, records)
        report, records = fixture()
        report["pairs"].pop()
        with self.assertRaisesRegex(TrainingGateError, "exact record/report alignment"):
            selected_pairs(report, records)
        report, records = fixture()
        with self.assertRaisesRegex(TrainingGateError, "exact record/report alignment"):
            selected_pairs(report, records[:-1])

    def test_nonfinite_quality_rejected_in_any_split(self):
        for state in ("repair", "dev"):
            for value in (float("nan"), float("inf"), -float("inf"), True):
                report, records = fixture()
                pair(report, state, "prompt_repair")["hps_delta"] = value
                with self.assertRaisesRegex(TrainingGateError, "Nonfinite"):
                    selected_pairs(report, records)
        report, records = fixture()
        pair(report, "repair", "prompt_repair")["candidate_metrics"]["siglip"] = float("nan")
        with self.assertRaisesRegex(TrainingGateError, "Nonfinite"):
            selected_pairs(report, records)

    def test_missing_required_schema_and_delta_arithmetic_rejected(self):
        for field in ("hps_delta", "pick_delta", "baseline_metrics", "split"):
            report, records = fixture()
            del pair(report, "repair", "prompt_repair")[field]
            with self.assertRaisesRegex(TrainingGateError, "Missing"):
                selected_pairs(report, records)
        report, records = fixture()
        pair(report, "repair", "prompt_repair")["hps_delta"] = .9
        with self.assertRaisesRegex(TrainingGateError, "arithmetic mismatch"):
            selected_pairs(report, records)
        report, records = fixture()
        pair(report, "repair", "prompt_repair")["hps_delta"] = None
        with self.assertRaisesRegex(TrainingGateError, "Missing quality/delta inconsistency"):
            selected_pairs(report, records)

    def test_paired_fingerprint_and_report_split_mismatch(self):
        report, records = fixture()
        records[2]["state_fingerprint"] = "different-state"
        with self.assertRaisesRegex(TrainingGateError, "Paired state_fingerprint mismatch"):
            selected_pairs(report, records)
        report, records = fixture()
        pair(report, "repair", "prompt_repair")["state_fingerprint"] = "different-state"
        with self.assertRaisesRegex(TrainingGateError, "fingerprint mismatch"):
            selected_pairs(report, records)
        report, records = fixture()
        pair(report, "repair", "prompt_repair")["split"] = "val"
        with self.assertRaisesRegex(TrainingGateError, "Report/action split mismatch"):
            selected_pairs(report, records)

    def test_prompt_partition_leakage_rejected(self):
        report, records = fixture()
        for row in records:
            if row["state_id"] == "dev":
                row["prompt"] = "  ORIGINAL   PROMPT repair  "
        with self.assertRaisesRegex(TrainingGateError, "Prompt split leakage"):
            selected_pairs(report, records)

    def test_zero_gate_and_record_zero_require_exactness(self):
        report, records = fixture()
        report["matched_zero_intervention_passed"] = False
        with self.assertRaisesRegex(TrainingGateError, "zero intervention gate"):
            selected_pairs(report, records)
        report, records = fixture()
        records[2]["zero_intervention_test"]["latent_exact"] = False
        with self.assertRaisesRegex(TrainingGateError, "Matched zero intervention failed"):
            selected_pairs(report, records)

    def test_semantic_missing_and_negative_flip_not_teachers(self):
        for candidate in (None, 0):
            report, records = fixture()
            for action in ("conditional_revisit", "prompt_repair"):
                row = pair(report, "repair", action)
                row["candidate_correct"] = candidate
                row["candidate_metrics"]["semantic"] = candidate
            with self.assertRaisesRegex(TrainingGateError, "no train wrong-to-correct"):
                selected_pairs(report, records)
        report, records = fixture()
        # A harmful continuation on a correct baseline is not a teacher, but the
        # original correct state remains a valid preservation target.
        for action in ("conditional_revisit", "prompt_repair"):
            row = pair(report, "preserve", action)
            row["candidate_correct"] = 0
            row["candidate_metrics"]["semantic"] = 0
        self.assertEqual(preservation_states(report, records), ["preserve"])
        self.assertEqual(set(selected_pairs(report, records)), {"repair"})

    def test_conflicting_baseline_measurements_rejected(self):
        report, records = fixture()
        row = pair(report, "repair", "prompt_repair")
        row["baseline_metrics"] = dict(row["baseline_metrics"])
        row["baseline_correct"] = 1
        row["baseline_metrics"]["semantic"] = 1
        with self.assertRaisesRegex(TrainingGateError, "Conflicting baseline semantic"):
            selected_pairs(report, records)
        report, records = fixture()
        row = pair(report, "repair", "prompt_repair")
        row["baseline_metrics"] = dict(row["baseline_metrics"])
        row["baseline_metrics"]["hps"] += .001
        row["hps_delta"] = row["candidate_metrics"]["hps"] - row["baseline_metrics"]["hps"]
        with self.assertRaisesRegex(TrainingGateError, "Conflicting baseline quality"):
            selected_pairs(report, records)

    def test_selected_target_path_required(self):
        report, records = fixture()
        del records[2]["final_latent"]
        with self.assertRaisesRegex(TrainingGateError, "no final_latent path"):
            selected_pairs(report, records)

    def test_oracle_v2_teacher_interface_notes_do_not_change_quality_gate(self):
        for action in ("layout_plan", "opposite_layout", "regional_flow", "opposite_regional"):
            report, records = fixture()
            for row in records + report["pairs"]:
                if row["action"] == "prompt_repair":
                    row["action"] = action
            report["candidate_actions"] = [action if name == "prompt_repair" else name
                                           for name in report["candidate_actions"]]
            selected = selected_pairs(report, records)["repair"]
            self.assertEqual(selected["action"], action)
            note = selected["training_teacher_note"]
            if "regional" in action:
                self.assertIn("Offline regional velocity teacher", note)
                self.assertIn("original prompt", note)
            else:
                self.assertIn("Offline prompt-edit teacher", note)
            self.assertIn("not evidence", note)
            set_quality(report, "repair", action, "hps", .28)
            self.assertEqual(selected_pairs(report, records)["repair"]["action"], "conditional_revisit")


if __name__ == "__main__":
    unittest.main()
