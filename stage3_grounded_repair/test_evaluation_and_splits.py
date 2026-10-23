"""Hermetic stdlib tests; synthetic fixtures are not model experiment results.

Run: python -m unittest discover -s outputs/stage3_grounded_repair
     -p test_evaluation_and_splits.py -v
"""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import evaluate_repair_pairwise as evaluation
import make_prompt_splits as splitting


def fixture():
    records, gen, scores = [], [], []
    zero = {"passed": True, "pixel_max_abs_diff": 0, "latent_exact": True}
    for state, prompt, b, c in [("s0", "p0", False, True), ("s1", "p1", True, False),
                                ("s2", "p0", False, False), ("s3", "p2", None, True)]:
        for action, correct in (("baseline", b), ("repair", c)):
            key = {"state_id": state, "action": action}
            records.append(dict(key, prompt_id=prompt, seed=17, flow_index=30, true_t=.345,
                                split="architecture_dev", state_fingerprint=state + "-fingerprint",
                                image_path=f"/images/{state}_{action}.png", zero_intervention_test=zero.copy(),
                                cost={"visual_block_calls": 490 if action == "baseline" else 510,
                                      "mmu_checks": 0 if action == "baseline" else 1}))
            gen.append(dict(key, correct=correct, tag="position"))
            gain = .02 if state == "s2" and action == "repair" else 0
            scores.append(dict(key, pickscore=22. + gain, hps=.29 + gain, siglip=.2))
    return records, gen, scores


class PairwiseEvaluationTests(unittest.TestCase):
    def test_transition_counts_missing_and_costs(self):
        report = evaluation.evaluate(*fixture(), bootstrap_samples=100)
        summary = report["by_action"]["repair"]
        self.assertEqual(summary["wrong_to_correct"], 1)
        self.assertEqual(summary["correct_to_wrong"], 1)
        self.assertEqual(summary["classification_counts"], {
            "positive_repair": 1, "negative_flip": 1, "quality_only_gain": 1, "semantic_missing": 1})
        self.assertEqual(summary["metrics"]["semantic"]["n_missing_pairs"], 1)
        self.assertEqual(summary["metrics"]["semantic"]["n_prompts"], 2)
        self.assertEqual(summary["cost_metrics"]["visual_block_calls"]["baseline_mean"], 490)
        self.assertEqual(summary["cost_metrics"]["visual_block_calls"]["paired_delta"], 20)
        self.assertFalse(report["controller_gate"]["automatically_authorized"])

    def test_quality_loss_not_full_repair(self):
        records, gen, scores = fixture()
        scores[1]["hps"] -= .01
        report = evaluation.evaluate(records, gen, scores, bootstrap_samples=0)
        self.assertEqual(report["pairs"][0]["classification"], "semantic_gain_quality_loss")
        self.assertEqual(report["offline_oracle"]["quality_guarded_semantic"]["paired_delta"], 0)
        report = evaluation.evaluate(records, gen, scores, bootstrap_samples=0, hps_loss_tolerance=.011)
        self.assertEqual(report["pairs"][0]["classification"], "positive_repair")

    def test_unknown_quality_not_complete_success(self):
        records, gen, scores = fixture()
        scores[1]["hps"] = None
        report = evaluation.evaluate(records, gen, scores, bootstrap_samples=0)
        self.assertEqual(report["pairs"][0]["classification"], "semantic_gain_quality_unknown")
        self.assertEqual(report["by_action"]["repair"]["metrics"]["hps"]["n_missing_pairs"], 1)

    def test_zero_guard_requires_exact_test(self):
        for broken in ({"passed": False, "pixel_max_abs_diff": 0, "latent_exact": True},
                       {"passed": True, "pixel_max_abs_diff": 1, "latent_exact": True},
                       {"passed": True, "pixel_max_abs_diff": 0, "latent_exact": False},
                       {"passed": True, "pixel_max_abs_diff": 0}):
            records, gen, scores = fixture()
            records[1]["zero_intervention_test"] = broken
            with self.assertRaisesRegex(ValueError, "zero intervention"):
                evaluation.evaluate(records, gen, scores)

    def test_global_zero_test_coverage(self):
        records, gen, scores = fixture()
        for record in records:
            del record["zero_intervention_test"]
        test = {"passed": True, "pixel_max_abs_diff": 0, "latent_exact": True,
                "state_ids": [f"s{i}" for i in range(4)]}
        self.assertTrue(evaluation.evaluate(records, gen, scores, zero_test=test, bootstrap_samples=0)["matched_zero_intervention_passed"])
        test["state_ids"].pop()
        with self.assertRaisesRegex(ValueError, "cover"):
            evaluation.evaluate(records, gen, scores, zero_test=test)

    def test_reject_duplicate_and_missing_rows(self):
        for stream in range(3):
            arrays = fixture()
            arrays[stream].append(deepcopy(arrays[stream][0]))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                evaluation.evaluate(*arrays)
        for stream in (1, 2):
            arrays = fixture()
            arrays[stream].pop()
            with self.assertRaisesRegex(ValueError, "identical key"):
                evaluation.evaluate(*arrays)

    def test_reject_incomplete_action_set_and_state_mismatch(self):
        arrays = fixture()
        for array in arrays:
            array.pop()
        with self.assertRaisesRegex(ValueError, "baseline|Incomplete"):
            evaluation.evaluate(*arrays)
        for field in ("prompt_id", "seed", "flow_index", "state_fingerprint", "split"):
            arrays = fixture()
            arrays[0][1][field] = "changed"
            with self.assertRaisesRegex(ValueError, "pairing mismatch"):
                evaluation.evaluate(*arrays)

    def test_official_filename_mapping_and_duplicate_filename(self):
        records, gen, scores = fixture()
        official = [dict(filename=records[i]["image_path"], correct=row["correct"], tag=row["tag"])
                    for i, row in enumerate(gen)]
        self.assertEqual(evaluation.evaluate(records, official, scores, bootstrap_samples=0)["n_states"], 4)
        records[1]["image_path"] = records[0]["image_path"]
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            evaluation.evaluate(records, official, scores)

    def test_nonfinite_and_bad_correct_rejected(self):
        records, gen, scores = fixture()
        scores[1]["hps"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluation.evaluate(records, gen, scores)
        records, gen, scores = fixture()
        gen[1]["correct"] = "unknown"
        with self.assertRaisesRegex(ValueError, "GenEval correct"):
            evaluation.evaluate(records, gen, scores)

    def test_cluster_bootstrap_keeps_prompt_group(self):
        # Twenty states from one prompt must move as one cluster.
        rows = [("many", 0, 1)] * 20 + [("one", 1, 0)]
        result = evaluation.cluster_summary(rows, 2000, 3)
        self.assertEqual(result["n_prompts"], 2)
        self.assertEqual(result["prompt_cluster_bootstrap_95pct_ci"], [-1., 1.])
        single = evaluation.cluster_summary(rows[:20], 100, 3)
        self.assertIsNone(single["prompt_cluster_bootstrap_95pct_ci"])

    def test_individual_oracle_not_joint_claim(self):
        records, gen, scores = fixture()
        report = evaluation.evaluate(records, gen, scores, bootstrap_samples=0)
        oracle = report["offline_oracle"]
        self.assertTrue(oracle["metrics_choose_different_actions"])
        self.assertEqual(oracle["per_metric"]["semantic"]["paired_delta"], 1 / 3)
        self.assertIn("offline upper bound", oracle["label"])

    def test_cli_writes_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = []
            for name, rows in zip(("records", "geneval", "scores"), fixture()):
                path = directory / (name + ".jsonl")
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                inputs.extend(["--" + name, str(path)])
            output, pairs = directory / "report.json", directory / "pairs.jsonl"
            evaluation.main(inputs + ["--output", str(output), "--paired-output", str(pairs), "--bootstrap-samples", "10"])
            self.assertEqual(len(evaluation.read_jsonl(pairs)), 4)
            self.assertTrue(output.is_file())


def metadata(count=300):
    return [{"prompt": f"A distinct relation prompt {i}", "tag": ("position" if i % 2 else "counting"),
             "include": [{"class": "bus", "count": 1}]} for i in range(count)]


class PromptSplitTests(unittest.TestCase):
    def test_disjoint_reproducible_full_splits(self):
        items = metadata()
        a = splitting.make_splits(items, excluded_prompts=[items[55]["prompt"]])
        b = splitting.make_splits(items, excluded_prompts=[items[55]["prompt"]])
        self.assertEqual(a, b)
        all_rows = [row for name in splitting.SPLITS for row in a[name]]
        hashes = [row["prompt_hash"] for row in all_rows]
        self.assertEqual(len(hashes), len(set(hashes)))
        texts = {splitting.normalize_prompt(row["prompt"]) for row in all_rows}
        self.assertFalse(texts & splitting.old_geneval_subset(items))
        self.assertNotIn(splitting.normalize_prompt(items[55]["prompt"]), texts)
        self.assertEqual(len(a["train"]), 120)
        self.assertEqual(len(a["val"]), 40)
        self.assertTrue(a["sealed_test"])

    def test_normalized_prompt_cannot_leak(self):
        items = metadata(100)
        items.append(dict(items[0], prompt="  A   DISTINCT relation prompt 0  "))
        splits = splitting.make_splits(items, architecture_dev=10, train=30, val=10,
                                       test_min=10, legacy_per_category=0)
        rows = [row for name in splitting.SPLITS for row in splits[name]]
        self.assertEqual(len(rows), 100)
        self.assertEqual(splitting.normalize_prompt("Ａ cat"), splitting.normalize_prompt("A  CAT"))

    def test_shortage_does_not_silently_reuse_test(self):
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            splitting.make_splits(metadata(60), legacy_per_category=0)
        splits = splitting.make_splits(metadata(60), legacy_per_category=0, allow_reduced=True)
        self.assertTrue(splits["_provenance"]["reduced_partition"])
        self.assertFalse(splits["_provenance"]["train_target_range_met"])
        self.assertGreater(len(splits["sealed_test"]), 0)
        self.assertEqual(sum(len(splits[name]) for name in splitting.SPLITS), 60)

    def test_external_train_keeps_independent_val_and_test(self):
        extra = [{"prompt": f"External training example {i}", "tag": "position"} for i in range(120)]
        splits = splitting.make_splits(metadata(100), additional_train=extra, legacy_per_category=0)
        self.assertEqual(len(splits["train"]), 120)
        self.assertTrue(all(row["source"] == "additional_train" for row in splits["train"]))
        self.assertTrue(all(row["source"] == "official" for row in splits["val"] + splits["sealed_test"]))
        with self.assertRaisesRegex(ValueError, "overlaps"):
            splitting.make_splits(metadata(), additional_train=[metadata()[3]])

    def test_smoke_does_not_access_validation_or_test(self):
        class Restricted(dict):
            def get(self, key, default=None):
                if key in ("sealed_test", "val"):
                    raise AssertionError("Sealed or validation data read during smoke")
                return super().get(key, default)
            def __getitem__(self, key):
                if key in ("sealed_test", "val"):
                    raise AssertionError("Sealed or validation data read during smoke")
                return super().__getitem__(key)
        safe = Restricted(architecture_dev=[{"prompt_id": "dev", "split": "architecture_dev"}],
                          train=[{"prompt_id": "train", "split": "train"}])
        self.assertEqual([row["prompt_id"] for row in splitting.select_smoke(safe, 2)], ["dev", "train"])
        with self.assertRaisesRegex(ValueError, "Not enough"):
            splitting.select_smoke(safe, 3)

    def test_writing_sealed_manifest_and_no_overwrite(self):
        splits = splitting.make_splits(metadata(), legacy_per_category=12)
        with tempfile.TemporaryDirectory() as directory:
            manifest = splitting.write_splits(splits, directory)
            self.assertNotIn("category_counts", manifest["artifacts"]["sealed_test"])
            self.assertEqual(set(manifest["artifacts"]["sealed_test"]), {"file", "prompts", "sha256"})
            sealed = Path(directory) / "sealed_test.jsonl"
            self.assertTrue(sealed.is_file())
            with self.assertRaises(FileExistsError):
                splitting.write_splits(splits, directory)

    def test_history_reader_and_tag_filter_after_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            path.write_text(json.dumps({"train": [{"prompt": "Old sample"}], "val": [{"prompt": "Other"}]}), encoding="utf-8")
            self.assertEqual(splitting.read_excluded_prompts(path), ["Old sample", "Other"])
        items = metadata()
        splits = splitting.make_splits(items, tags={"position"}, architecture_dev=10, train=60,
                                       val=30, test_min=10)
        all_rows = [row for name in splitting.SPLITS for row in splits[name]]
        self.assertTrue(all(row["tag"] == "position" for row in all_rows))
        self.assertEqual(len(all_rows), 138)  # 150 position - historical 12.

    def test_cli_explicit_remote_history_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            history = directory / "remote_work_prompts.json"
            official = directory / "metadata.jsonl"
            history.write_text(json.dumps([{ "prompt": "A distinct relation prompt 55"}]), encoding="utf-8")
            official.write_text("".join(json.dumps(row) + "\n" for row in metadata()), encoding="utf-8")
            splitting.main(["--metadata", str(official), "--history-stage1", str(history),
                            "--output-dir", str(directory / "splits")])
            manifests = [splitting.read_metadata(directory / "splits" / (name + ".jsonl"))
                         for name in splitting.SPLITS]
            self.assertFalse(any(row["prompt"] == "A distinct relation prompt 55"
                                 for rows in manifests for row in rows))
            with self.assertRaisesRegex(FileNotFoundError, "history-stage1"):
                splitting.main(["--metadata", str(official), "--history-stage1", str(directory / "absent.json"),
                                "--output-dir", str(directory / "missing")])


if __name__ == "__main__":
    unittest.main()
