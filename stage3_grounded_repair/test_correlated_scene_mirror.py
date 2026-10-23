import unittest
import torch
from correlated_scene_mirror import reflected_input


class CorrelatedMirrorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.x=torch.randn(1,3,4,4).repeat(2,1,1,1)
        self.v=torch.randn(1,3,4,4).repeat(2,1,1,1)
        self.noise=torch.randn(1,3,4,4).repeat(2,1,1,1)
        self.clean=self.x[:1]+.6*self.v[:1]

    def test_native_reflection_preserves_all_values_and_cfg(self):
        y=reflected_input(self.x,self.v,self.noise,None,.4,'latent_target')
        self.assertTrue(torch.equal(y.flip(-1),self.x))
        self.assertTrue(torch.equal(y[0],y[1]))

    def test_wrong_axis_is_distinct(self):
        y=reflected_input(self.x,self.v,self.noise,None,.4,'latent_wrong_axis')
        self.assertTrue(torch.equal(y,self.x.flip(-2)))
        self.assertFalse(torch.equal(y,self.x.flip(-1)))

    def test_inferred_co_reflection_matches_native_when_clean_is_exact(self):
        y=reflected_input(self.x,self.v,self.noise,self.clean.flip(-1),.4,'inferred_correlated_target')
        torch.testing.assert_close(y,self.x.flip(-1),rtol=1e-6,atol=1e-6)

    def test_seed_is_transformed_deterministically_not_resampled(self):
        desired=self.clean.flip(-1)
        y=reflected_input(self.x,self.v,self.noise,desired,.4,'seed_correlated_target')
        expected=.6*self.noise.flip(-1)+.4*desired.repeat(2,1,1,1)
        torch.testing.assert_close(y,expected)
        self.assertTrue(torch.equal(y,reflected_input(self.x,self.v,self.noise,desired,.4,'seed_correlated_target')))

    def test_rejects_invalid_mode_and_cfg(self):
        with self.assertRaises(ValueError):reflected_input(self.x,self.v,self.noise,self.clean,.4,'unknown')
        bad=self.x.clone();bad[1,0,0,0]+=1
        with self.assertRaises(ValueError):reflected_input(bad,self.v,self.noise,self.clean,.4,'latent_target')


if __name__=='__main__':unittest.main()
