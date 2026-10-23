import unittest
import torch
from build_pixel_teacher_candidates import geometry_gate, clean_import_modifier


class TeacherContinuationTests(unittest.TestCase):
    def test_gate_only_intermediate_target_and_missing_rejected(self):
        actions = ('original_clean', 'original_roundtrip', 'target_pixels', 'target_roundtrip',
                   'opposite_pixels', 'opposite_roundtrip')
        rows = [{'state_id': str(i), 'action': a, 'correct': i < 4 and a.startswith('target')}
                for i in range(6) for a in actions]
        self.assertEqual(sum(geometry_gate({'rows': rows}).values()), 4)
        with self.assertRaises(ValueError):
            geometry_gate({'rows': rows[:-1]})
        with self.assertRaises(ValueError):
            geometry_gate({'rows': rows + [rows[0]]})

    def test_clean_import_preserves_noise_residual_and_cfg(self):
        grid = torch.linspace(0, 1, 50)
        state = {'time_grid': grid}
        t = grid[20]
        noise = torch.randn(1, 16, 4, 4)
        clean = torch.randn_like(noise)
        desired = torch.randn_like(noise)
        x_one = (1-t)*noise + t*clean
        x = torch.cat((x_one, x_one))
        velocity = torch.cat((clean-noise, clean-noise))
        modified, _, _ = clean_import_modifier(state, desired)(None, x, t, velocity)
        dt = grid[21]-grid[20]
        moved = x + dt*(modified-velocity)
        torch.testing.assert_close(moved[:1]-t*desired, x[:1]-t*clean)
        torch.testing.assert_close(moved[0], moved[1])


if __name__ == '__main__':
    unittest.main()
