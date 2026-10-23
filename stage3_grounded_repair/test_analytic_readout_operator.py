"""Small conversion/routing tests for the new inference boundary."""
import types
import unittest
import torch
from analytic_readout_operator import converted_readouts, recurrence_mode, READOUT_KEYS


class AnalyticReadoutTests(unittest.TestCase):
    def adapter(self):
        return {'outer.readout.weight': torch.zeros(4, 6), 'inner.readout.weight': torch.zeros(4, 8),
            'outer.readout.bias': torch.zeros(4), 'inner.readout.bias': torch.zeros(4),
            'outer.feedback.weight': torch.randn(3, 2)}

    def test_split_and_common_bias_prediction(self):
        torch.manual_seed(19)
        adapter = self.adapter(); coefficient = torch.randn(15, 4, dtype=torch.float64)
        derived, dims = converted_readouts(adapter, coefficient)
        outer = torch.randn(7, 6); inner = torch.randn(7, 8)
        reference = torch.cat((outer.double(), inner.double(), torch.ones(7, 1).double()), -1) @ coefficient
        actual = (outer @ derived['outer.readout.weight'].T + derived['outer.readout.bias'] +
                  inner @ derived['inner.readout.weight'].T + derived['inner.readout.bias'])
        torch.testing.assert_close(actual.double(), reference, rtol=1e-5, atol=1e-6)
        self.assertEqual(dims['outer_input_dimensions'], 6)
        self.assertTrue(torch.equal(derived['outer.feedback.weight'], adapter['outer.feedback.weight']))
        self.assertEqual(set(k for k in derived if not torch.equal(derived[k], adapter[k])),
                         set(READOUT_KEYS)-{'inner.readout.bias'})

    def test_bad_coefficient_rejected(self):
        for bad in (torch.zeros(14, 4, dtype=torch.float64), torch.zeros(15, 4),
                    torch.full((15, 4), float('nan'), dtype=torch.float64)):
            with self.assertRaises(ValueError):
                converted_readouts(self.adapter(), bad)

    def test_no_loop_really_skips_cores_and_restores(self):
        class Branch:
            def run_core(self, index, hidden, kwargs):
                raise AssertionError('Shared core must not run')
        op = types.SimpleNamespace(outer=Branch(), inner=Branch())
        original = op.outer.run_core; hidden = torch.randn(2, 3)
        with recurrence_mode(op, 'no_loop') as counts:
            self.assertIs(op.outer.run_core(0, hidden, {}), hidden)
            self.assertIs(op.inner.run_core(1, hidden, {}), hidden)
        self.assertEqual(counts, dict(outer=0, inner=0))
        self.assertEqual(op.outer.run_core, original)

    def test_loop_counts_actual_calls_and_restores_on_error(self):
        class Branch:
            def run_core(self, index, hidden, kwargs):
                return hidden + index + 1
        op = types.SimpleNamespace(outer=Branch(), inner=Branch())
        original = op.inner.run_core
        with self.assertRaisesRegex(RuntimeError, 'probe'):
            with recurrence_mode(op, 'loop2') as counts:
                self.assertEqual(op.outer.run_core(1, 3, {}), 5)
                self.assertEqual(op.inner.run_core(0, 3, {}), 4)
                raise RuntimeError('probe')
        self.assertEqual(counts, dict(outer=1, inner=1))
        self.assertEqual(op.inner.run_core, original)


if __name__ == '__main__':
    unittest.main()
