import unittest
from types import SimpleNamespace
import torch
from torch import nn
from runtime import tensor_hash
from fm_rebridge import linear_rebridge, continue_rebridged


class BridgeTests(unittest.TestCase):
    def test_linear_endpoints_float32_and_cfg(self):
        one = torch.randn(1, 3, 4, 4).bfloat16()
        noise = torch.cat((one, one))
        clean = torch.randn(1, 3, 4, 4)
        torch.testing.assert_close(linear_rebridge(noise, clean, 0), noise.float())
        torch.testing.assert_close(linear_rebridge(noise, clean, 1)[:1], clean)
        bridge = linear_rebridge(noise, clean, .45)
        self.assertEqual(bridge.dtype, torch.float32)
        torch.testing.assert_close(bridge[0], bridge[1])

    def test_first_velocity_full_and_original_seed_used(self):
        noise = torch.ones(2, 1, 2, 2)
        clean = torch.zeros(1, 1, 2, 2)
        grid = torch.tensor([0., .5, 1.])
        observed = []
        class FakeRuntime:
            device = 'cpu'
            cfg = 5.
            config = SimpleNamespace(transport=SimpleNamespace(path_type='Linear'))
            model = nn.Module()
            original = [nn.Identity() for _ in range(10)]
            def noise(self, seed):
                self.used_seed = seed
                return noise.clone()
            def inputs(self, prompt): return {}
            def velocity(self, x, t, kwargs):
                observed.append(x.clone())
                return -torch.ones_like(x)
        rt = FakeRuntime()
        rt.grid = grid
        prefix = torch.zeros(1)
        state = {'time_grid': grid, 'CFG': 5., 'flow_index': 1, 'true_t': .5, 'seed': 101,
                 'prefix_cache': prefix, 'prefix_cache_hash': tensor_hash(prefix), 'prompt': 'example',
                 'runtime': {'base_model_hash': 'a596cbc305c1df987c125d4f218e78f39b681621904cccfb2a3bf0ca0327f92c'},
                 'latent': noise.clone(), 'baseline_prediction': -noise.clone()}
        final, cost, diag = continue_rebridged(rt, state, clean)
        self.assertEqual(rt.used_seed, 101)
        torch.testing.assert_close(observed[0], torch.ones_like(noise)*.5)
        torch.testing.assert_close(final, torch.zeros_like(noise))
        self.assertEqual(cost['visual_block_calls'], 10)
        self.assertTrue(diag['first_velocity_recomputed'])

    def test_nonidentical_cfg_rejected(self):
        with self.assertRaises(ValueError):
            linear_rebridge(torch.stack((torch.zeros(1,2,2), torch.ones(1,2,2))), torch.zeros(1,1,2,2), .5)


if __name__ == '__main__':
    unittest.main()
