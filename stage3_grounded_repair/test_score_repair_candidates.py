"""Hermetic alignment/hash/merge tests; fixture values are not model results."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import score_repair_candidates as scoring


def fixture(directory):
    rows = []
    for index in range(2):
        for action in ("baseline", "prompt_repair"):
            path = directory / f"state{index}_{action}.png"
            path.write_bytes(f"synthetic-bytes-{index}-{action}".encode())
            rows.append({"state_id": f"state{index}", "action": action,
                         "prompt_id": f"prompt{index}", "prompt": f"Original prompt {index}",
                         "seed": 101, "flow_index": 30, "true_t": .345,
                         "state_fingerprint": f"state-hash-{index}", "split": "architecture_dev",
                         "generation_prompt": "Different teacher rewrite" if action == "prompt_repair" else None,
                         "image_path": str(path)})
    return rows


def streams(records):
    pick = scoring.prepare_score_rows(records)
    hps = scoring.prepare_score_rows(records)
    siglip = scoring.prepare_score_rows(records)
    for index, row in enumerate(pick):
        row["pickscore"] = 22. + index / 100
        row["metric_models"]["pickscore"] = "/models/pickscore-v1"
        row["score_definitions"]["pickscore"] = scoring.SCORE_DEFINITIONS["pickscore"]
    for index, row in enumerate(hps):
        row["hps"] = .29 + index / 1000
    for index, row in enumerate(siglip):
        row["siglip"] = .2 + index / 1000
    return pick, hps, siglip


class ScoreMergeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.records = fixture(self.directory)

    def tearDown(self):
        self.temporary.cleanup()

    def test_original_prompt_teacher_override_and_hash(self):
        rows = scoring.prepare_score_rows(self.records)
        for row in rows:
            original = f"Original prompt {row['state_id'][-1]}"
            self.assertEqual(row["scored_prompt"], original)
            self.assertEqual(row["scored_prompt_sha256"], hashlib.sha256(original.encode()).hexdigest())
            self.assertEqual(row["image_sha256"], scoring.file_sha256(row["image_path"]))
            self.assertTrue(all(row[metric] is None for metric in scoring.METRICS))

    def test_strict_complete_merge(self):
        output = scoring.merge_score_rows(self.records, streams(self.records), require_metrics=scoring.METRICS)
        self.assertEqual(len(output), 4)
        self.assertTrue(all(all(row[metric] is not None for metric in scoring.METRICS) for row in output))
        self.assertEqual(output[0]["metric_models"]["pickscore"], "/models/pickscore-v1")

    def test_uncomputed_stays_null_and_required_rejected(self):
        pick, _, _ = streams(self.records)
        output = scoring.merge_score_rows(self.records, [pick])
        self.assertTrue(all(row["hps"] is None and row["siglip"] is None for row in output))
        with self.assertRaisesRegex(ValueError, "Required metrics missing"):
            scoring.merge_score_rows(self.records, [pick], require_metrics=("hps",))

    def test_missing_duplicate_extra_rows_rejected(self):
        pick, hps, _ = streams(self.records)
        with self.assertRaisesRegex(ValueError, "missing or extra"):
            scoring.merge_score_rows(self.records, [pick[:-1], hps])
        with self.assertRaisesRegex(ValueError, "Duplicate score stream"):
            scoring.merge_score_rows(self.records, [pick + [deepcopy(pick[0])]])
        extra = deepcopy(pick[0]); extra["state_id"] = "unexpected"
        with self.assertRaisesRegex(ValueError, "missing or extra"):
            scoring.merge_score_rows(self.records, [pick + [extra]])

    def test_duplicate_nonnull_metric_rejected_even_when_equal(self):
        pick, _, _ = streams(self.records)
        with self.assertRaisesRegex(ValueError, "Duplicate nonnull metric"):
            scoring.merge_score_rows(self.records, [pick, deepcopy(pick)])

    def test_missing_field_not_silently_null(self):
        pick, _, _ = streams(self.records)
        del pick[0]["hps"]
        with self.assertRaisesRegex(ValueError, "Explicit hps:null"):
            scoring.merge_score_rows(self.records, [pick])

    def test_image_change_after_scoring_rejected(self):
        pick, _, _ = streams(self.records)
        Path(self.records[0]["image_path"]).write_bytes(b"replacement")
        with self.assertRaisesRegex(ValueError, "image_sha256 mismatch"):
            scoring.merge_score_rows(self.records, [pick])

    def test_wrong_scoring_prompt_rejected(self):
        pick, _, _ = streams(self.records)
        pick[0]["scored_prompt"] = "teacher reformulation"
        with self.assertRaisesRegex(ValueError, "scored prompt mismatch"):
            scoring.merge_score_rows(self.records, [pick])
        pick[0]["scored_prompt_sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "scored_prompt_sha256 mismatch"):
            scoring.merge_score_rows(self.records, [pick])
        self.records[1]["prompt"] = "teacher reformulation"
        with self.assertRaisesRegex(ValueError, "Original scoring prompt mismatch"):
            scoring.prepare_score_rows(self.records)

    def test_missing_images_fail(self):
        Path(self.records[0]["image_path"]).unlink()
        with self.assertRaises(FileNotFoundError):
            scoring.prepare_score_rows(self.records)

    def test_nonfinite_and_bool_score_rejected(self):
        for value in (float("nan"), float("inf"), True):
            pick, _, _ = streams(self.records)
            pick[0]["pickscore"] = value
            with self.assertRaisesRegex(ValueError, "finite number"):
                scoring.merge_score_rows(self.records, [pick])

    def test_action_shards_complete_and_duplicate_files(self):
        for index in range(2):
            scoring.write_jsonl(self.directory / f"actions_shard{index}.jsonl", self.records[index * 2:index * 2 + 2])
        self.assertEqual(len(scoring.load_action_records([self.directory])), 4)
        with self.assertRaisesRegex(ValueError, "same action-record file"):
            scoring.load_action_records([self.directory, self.directory / "actions_shard0.jsonl"])
        scoring.write_jsonl(self.directory / "actions_shard2.jsonl", [self.records[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate action records"):
            scoring.load_action_records([self.directory])

    def test_nested_launcher_shards_ignore_aggregates(self):
        for index in range(2):
            shard = self.directory / f"shard{index}" / "actions_shard0.jsonl"
            scoring.write_jsonl(shard, self.records[index * 2:index * 2 + 2])
        # The pipeline writes this aggregate beside shards. It must not be read
        # again on scoring/merge, which would create duplicate keys.
        scoring.write_jsonl(self.directory / "all_actions.jsonl", self.records)
        scoring.write_jsonl(self.directory / "scores.jsonl", streams(self.records)[0])
        rows = scoring.load_action_records([self.directory])
        self.assertEqual(len(rows), 4)
        paths = scoring.expand_record_paths([self.directory])
        self.assertEqual(len(paths), 2)
        self.assertTrue(all(path.name == "actions_shard0.jsonl" for path in paths))
        quoted_glob = str(self.directory / "**" / "actions_shard*.jsonl")
        self.assertEqual(len(scoring.load_action_records([quoted_glob])), 4)
        with self.assertRaisesRegex(ValueError, "same action-record file"):
            scoring.load_action_records([self.directory, self.directory / "shard0"])

    def test_dev12x2_launcher_shape_and_terminal_report(self):
        import evaluate_repair_pairwise as paired
        # 12 prompts * 2 seeds * 2 fork indices * 8 actions = 384 synthetic
        # records across seven worker directories; no model outputs claimed.
        names = ("baseline", "generic_revisit", "extra_ode", "conditional_zero",
                 "semantic_probe", "opposite_probe", "prompt_repair", "opposite_prompt")
        partitions = [[] for _ in range(7)]
        official, quality = [], []
        state_counter = 0
        zero = {"passed": True, "latent_exact": True, "pixel_max_abs_diff": 0}
        for prompt in range(12):
            for seed in (101, 102):
                for flow_index in (30, 35):
                    state_id = f"{prompt:05d}_seed{seed}_i{flow_index}"
                    for action in names:
                        path = self.directory / "images" / action / state_id / "samples" / "0000.png"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(f"synthetic fixture {state_id} {action}".encode())
                        record = {"state_id": state_id, "action": action, "prompt_id": f"{prompt:05d}",
                                  "prompt": f"Original spatial prompt {prompt}", "seed": seed,
                                  "flow_index": flow_index, "true_t": .345 if flow_index == 30 else .455,
                                  "split": "architecture_dev", "state_fingerprint": state_id,
                                  "image_path": str(path), "zero_intervention_test": dict(zero)}
                        partitions[state_counter % 7].append(record)
                        # Include strict quality-preserving and quality-cost
                        # semantic repairs so extraction cannot conflate them.
                        correct = action in ("semantic_probe", "prompt_repair")
                        official.append({"filename": str(path), "correct": correct, "tag": "position"})
                        quality.append({"state_id": state_id, "action": action, "pickscore": 22.,
                                        "hps": .28 if action == "semantic_probe" else .29, "siglip": .2})
                    state_counter += 1
        for worker, rows in enumerate(partitions):
            scoring.write_jsonl(self.directory / f"shard{worker}" / "actions_shard0.jsonl", rows)
        flattened = scoring.load_action_records([self.directory])
        self.assertEqual(len(flattened), 384)
        scoring.write_jsonl(self.directory / "all_actions.jsonl", flattened)
        self.assertEqual(len(scoring.load_action_records([self.directory])), 384)
        report = paired.evaluate(flattened, official, quality, bootstrap_samples=100)
        self.assertEqual(report["n_states"], 48)
        self.assertEqual(report["n_prompts"], 12)
        eligible = [pair for pair in report["pairs"]
                    if pair["baseline_correct"] == 0 and pair["candidate_correct"] == 1
                    and pair["hps_delta"] is not None and pair["hps_delta"] >= 0
                    and pair["pick_delta"] is not None and pair["pick_delta"] >= 0]
        self.assertEqual(len(eligible), 48)
        self.assertTrue(all(pair["action"] == "prompt_repair" for pair in eligible))
        self.assertNotIn("delta", eligible[0])  # Canonical report uses flat deltas.
        self.assertEqual(report["offline_oracle"]["quality_guarded_semantic"]["paired_delta"], 1)
        self.assertFalse(report["controller_gate"]["automatically_authorized"])

    def test_missing_official_metadata_remains_hard_error(self):
        from prepare_geneval_layout import prepare
        rows = deepcopy(self.records)
        for row in rows:
            path = self.directory / "missing_metadata" / row["action"] / row["state_id"] / "samples" / "0000.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic fixture")
            row["image_path"] = str(path)
        with self.assertRaises(FileNotFoundError) as caught:
            prepare(rows, self.directory / "official_layout")
        self.assertIn("metadata.jsonl", str(caught.exception))

    def test_incomplete_actions_or_pair_fingerprint_rejected(self):
        with self.assertRaisesRegex(ValueError, "baseline and a candidate|Incomplete action"):
            scoring.prepare_score_rows(self.records[:-1])
        self.records[1]["state_fingerprint"] = "another-state"
        with self.assertRaisesRegex(ValueError, "Same-state state_fingerprint mismatch"):
            scoring.prepare_score_rows(self.records)

    def test_cli_merge_without_torch(self):
        action_dir = self.directory / "actions"
        action_dir.mkdir()
        scoring.write_jsonl(action_dir / "actions_shard0.jsonl", self.records)
        score_paths = []
        for name, rows in zip(scoring.METRICS, streams(self.records)):
            path = self.directory / (name + ".jsonl")
            scoring.write_jsonl(path, rows)
            score_paths.append(str(path))
        output = self.directory / "merged.jsonl"
        scoring.main(["--records", str(action_dir), "--merge"] + score_paths +
                     ["--require-metrics", "pickscore,hps,siglip", "--output", str(output)])
        self.assertEqual(len(scoring.read_jsonl(output)), 4)


if __name__ == "__main__":
    unittest.main()
