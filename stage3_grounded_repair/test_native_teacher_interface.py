import unittest
import torch
from native_teacher_interface import physical, inferred_noise, rebridge, additive_parent, validate_identity


class NativeInterfaceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(68)
        self.x = torch.randn(1, 2, 3, 3).repeat(2, 1, 1, 1)
        self.v = torch.randn(1, 2, 3, 3).repeat(2, 1, 1, 1)
        self.t = torch.tensor(.45454540848731995)

    def test_linear_path_noise_and_endpoint(self):
        noise = inferred_noise(self.x, self.v, self.t)
        clean = self.x + (1-self.t)*self.v
        self.assertTrue(torch.allclose(rebridge(noise, clean, self.t), self.x, atol=3e-7))

    def test_bridge_target_difference_only(self):
        noise = inferred_noise(self.x, self.v, self.t)
        parent = self.x + self.v
        a, b = rebridge(noise, self.x, self.t), rebridge(noise, parent, self.t)
        self.assertTrue(torch.allclose(b-a, self.t*self.v, atol=3e-7))

    def test_additive_records_real_fp32_rounding(self):
        parent = self.x*.123
        value, delta = additive_parent(self.x, parent)
        self.assertTrue(torch.equal(value, self.x+(parent-self.x)))
        self.assertTrue(torch.equal(delta[0], delta[1]))
        self.assertLess(float((value-parent).abs().max()), 1e-6)

    def test_reject_invalid_physical_branches(self):
        bad = self.x.clone(); bad[1, 0, 0, 0] += 1
        with self.assertRaises(ValueError): physical(bad)
        with self.assertRaises(ValueError): physical(self.x.to(torch.bfloat16))
        bad = self.x.clone(); bad[:, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError): physical(bad)

    def test_split_and_pair_isolation(self):
        pair = dict(source_state_id='s', native_parent_state_id='p', source_seed=101,
                    parent_seed=202, split='train', prompt_id='a', prompt='a photo',
                    flow_index=35, true_t=float(self.t))
        common = {k: pair[k] for k in ('split','prompt_id','prompt','flow_index','true_t')}
        common.update(latent=self.x, baseline_final_latent=self.x, CFG=5.,
                      time_grid=self.t.repeat(50), runtime={'base_model_hash':'base'})
        source, parent = dict(common,state_id='s',seed=101), dict(common,state_id='p',seed=202)
        validate_identity(source, parent, pair)
        with self.assertRaises(ValueError): validate_identity(dict(source,split='val'),parent,pair)
        with self.assertRaises(ValueError): validate_identity(source,dict(parent,prompt='other'),pair)


if __name__ == '__main__': unittest.main()
