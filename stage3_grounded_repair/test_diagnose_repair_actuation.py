"""Tiny CPU-Torch snapshots; skip numerical tests when Torch is unavailable."""

import json
import math
from pathlib import Path
import tempfile
import unittest

import diagnose_repair_actuation as diagnostic

try:
    import torch
except ImportError:
    torch = None


class FileHashTests(unittest.TestCase):
    def test_existing_hash_verified_and_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, latent = root / "image.png", root / "latent.pt"
            image.write_bytes(b"image"); latent.write_bytes(b"snapshot")
            record = {"state_id": "state", "action": "repair", "image_path": str(image), "final_latent": str(latent),
                      "image_path_sha256": diagnostic._file_hash(image), "final_latent_sha256": diagnostic._file_hash(latent)}
            verified = diagnostic.verify_record_files(record)
            self.assertTrue(all(row["provenance_status"] == "stored_hash_verified" for row in verified.values()))
            image.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                diagnostic.verify_record_files(record)

    def test_legacy_unhashed_not_falsely_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("image.png", "latent.pt"):
                (root / name).write_bytes(b"legacy bytes")
            record = {"image_path": str(root / "image.png"), "final_latent": str(root / "latent.pt")}
            verified = diagnostic.verify_record_files(record)
            self.assertTrue(all(row["provenance_status"] == "legacy_unhashed" for row in verified.values()))
            with self.assertRaisesRegex(ValueError, "Missing stored hash"):
                diagnostic.verify_record_files(record, require_hashes=True)


@unittest.skipIf(torch is None, "Torch unavailable locally; execute these CPU tests on the server")
class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, changed=True, hashed=True, candidate_action="repair"):
        records = []
        for state_index in range(2):
            base = torch.ones(1, 1, 2, 2)
            for action in ("baseline", candidate_action):
                final = base + 1 if changed and action != "baseline" else base.clone()
                path = self.root / f"{state_index}_{action}.pt"
                image = self.root / f"{state_index}_{action}.png"
                torch.save({"state_id": f"state{state_index}", "action": action, "latent": final}, path)
                image.write_bytes(b"changed-image" if changed and action != "baseline" else b"same-image")
                record = {"state_id": f"state{state_index}", "action": action, "prompt_id": "one_prompt",
                          "seed": 101, "flow_index": 20 + state_index, "true_t": .2,
                          "split": "architecture_dev", "state_fingerprint": f"state{state_index}",
                          "final_latent": str(path), "image_path": str(image), "prompt": "original prompt"}
                if hashed:
                    record.update(final_latent_sha256=diagnostic._file_hash(path), image_path_sha256=diagnostic._file_hash(image))
                records.append(record)
        return records

    def test_known_delta_and_nonindependent_state_counts(self):
        result = diagnostic.diagnose(self.fixture())
        self.assertEqual(result["n_states"], 2)
        self.assertEqual(result["n_prompts"], 1)
        self.assertEqual(result["n_prompt_seed_groups"], 1)
        self.assertTrue(result["all_provenance_hashes_verified"])
        for pair in result["pairs"]:
            self.assertEqual(pair["relative_l2"], 1)
            self.assertEqual(pair["mse"], 1)
            self.assertEqual(pair["max_abs"], 1)
            self.assertFalse(pair["latent_exact"])
            self.assertFalse(pair["image_sha_equal"])
        self.assertIn("not independent", result["independence_note"])

    def test_all_zero_mark_and_optional_rejection(self):
        records = self.fixture(changed=False)
        result = diagnostic.diagnose(records)
        self.assertEqual(result["all_zero_noncontrol_actions"], ["repair"])
        self.assertEqual(result["by_action"]["repair"]["status"], "all_zero_actuation")
        self.assertTrue(all(pair["latent_exact"] and pair["image_sha_equal"] for pair in result["pairs"]))
        with self.assertRaisesRegex(ValueError, "All-zero actuation"):
            diagnostic.diagnose(records, reject_all_zero=True)

    def test_zero_control_expected_not_rejected(self):
        result = diagnostic.diagnose(self.fixture(changed=False, candidate_action="conditional_zero"), reject_all_zero=True)
        self.assertEqual(result["all_zero_noncontrol_actions"], [])
        self.assertEqual(result["by_action"]["conditional_zero"]["status"], "expected_zero_control")

    def test_payload_wrong_teacher_id_and_action_rejected(self):
        for field, wrong in (("state_id", "wrong_state"), ("action", "wrong_action")):
            records = self.fixture()
            row = records[1]
            payload = torch.load(row["final_latent"], weights_only=True)
            payload[field] = wrong
            torch.save(payload, row["final_latent"])
            row["final_latent_sha256"] = diagnostic._file_hash(row["final_latent"])
            with self.assertRaisesRegex(ValueError, "payload state_id/action mismatch"):
                diagnostic.diagnose(records)

    def test_latent_file_hash_tampering_rejected(self):
        records = self.fixture()
        row = records[1]
        Path(row["final_latent"]).write_bytes(b"corrupt snapshot")
        with self.assertRaisesRegex(ValueError, "Stored file hash mismatch"):
            diagnostic.diagnose(records)

    def test_old_unhashed_and_zero_denominator(self):
        records = self.fixture(hashed=False)
        base_row = records[0]
        torch.save({"state_id": base_row["state_id"], "action": "baseline", "latent": torch.zeros(1, 1, 2, 2)}, base_row["final_latent"])
        result = diagnostic.diagnose(records)
        self.assertFalse(result["all_provenance_hashes_verified"])
        self.assertTrue(all(pair["provenance_status"] == "legacy_unhashed" for pair in result["pairs"]))
        self.assertTrue(math.isfinite(result["pairs"][0]["relative_l2"]))
        with self.assertRaisesRegex(ValueError, "Missing stored hash"):
            diagnostic.diagnose(records, require_hashes=True)

    def test_nonfinite_shape_and_duplicate_fail(self):
        for tensor in (torch.tensor([float("nan")]), torch.zeros(3)):
            records = self.fixture()
            row = records[1]
            torch.save({"state_id": row["state_id"], "action": row["action"], "latent": tensor}, row["final_latent"])
            row["final_latent_sha256"] = diagnostic._file_hash(row["final_latent"])
            with self.assertRaisesRegex(ValueError, "nonfinite|shape mismatch"):
                diagnostic.diagnose(records)
        records = self.fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            diagnostic.diagnose(records + [records[0]])


if __name__ == "__main__":
    unittest.main()
