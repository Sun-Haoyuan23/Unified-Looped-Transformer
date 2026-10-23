import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from evaluate_oracle_pipeline import main


class EvaluationStartupTests(unittest.TestCase):
    def test_missing_experiment_writes_diagnostic_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch('sys.argv',['evaluate','--work',str(root),'--experiment','new_run']):
                with self.assertRaisesRegex(FileNotFoundError,'No action records'):
                    main()
            report=json.loads((root/'stage3_grounded_repair/new_run/evaluation_status.json').read_text())
            self.assertEqual(report['phase'],'failed')
            self.assertIn('No action records',report['error'])

    def test_waiting_startup_can_record_failure_before_generator_creates_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch('sys.argv',['evaluate','--work',str(root),'--experiment','new_run','--wait-generation']), \
                 patch('evaluate_oracle_pipeline.time.time',side_effect=[0.,0.,8000.,8000.]):
                with self.assertRaises(TimeoutError):main()
            report=json.loads((root/'stage3_grounded_repair/new_run/evaluation_status.json').read_text())
            self.assertEqual(report['phase'],'failed')
            self.assertIn('Generation did not finish',report['error'])


if __name__=='__main__':unittest.main()
