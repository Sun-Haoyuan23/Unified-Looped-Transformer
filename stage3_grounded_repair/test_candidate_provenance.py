import json
import tempfile
import unittest
from pathlib import Path
from candidate_provenance import file_hash, freeze_config, shuffled_source, verify_completed


class ProvenanceTests(unittest.TestCase):
    def test_resume_requires_identical_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            fingerprint = freeze_config(path, {'window': 4, 'source': 'abc'})
            self.assertEqual(fingerprint, freeze_config(path, {'source': 'abc', 'window': 4}, True))
            with self.assertRaises(RuntimeError): freeze_config(path, {'window': 5}, True)

    def test_legacy_records_cannot_be_silently_adopted(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                freeze_config(Path(directory) / 'config.json', {}, True)

    def test_resume_checks_both_file_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            image, latent = Path(directory) / 'a.png', Path(directory) / 'a.pt'
            image.write_bytes(b'png'); latent.write_bytes(b'latent')
            row = {'configuration_sha256': 'abc', 'image_path': str(image),
                'final_latent': str(latent), 'image_path_sha256': file_hash(image),
                'final_latent_sha256': file_hash(latent)}
            verify_completed(row, 'abc')
            latent.write_bytes(b'changed')
            with self.assertRaises(RuntimeError): verify_completed(row, 'abc')

    def test_shuffle_matches_split_and_step_and_excludes_original_prompt(self):
        rows = {'a': {'split': 'val', 'flow_index': 30, 'prompt_id': 'p1'},
                'same': {'split': 'val', 'flow_index': 30, 'prompt_id': 'p1'},
                'wrong_step': {'split': 'val', 'flow_index': 35, 'prompt_id': 'p2'},
                'wrong_split': {'split': 'train', 'flow_index': 30, 'prompt_id': 'p2'},
                'right': {'split': 'val', 'flow_index': 30, 'prompt_id': 'p3'}}
        self.assertEqual(shuffled_source(rows, 'a'), 'right')
        del rows['right']
        with self.assertRaises(RuntimeError): shuffled_source(rows, 'a')


if __name__ == '__main__': unittest.main()
