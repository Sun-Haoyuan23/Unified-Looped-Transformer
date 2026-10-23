import unittest
import torch
from rms_scaled_revisit import make_operator
from two_level_operator import make_operator as original_operator
import test_two_level_operator as helpers


class RMSTests(unittest.TestCase):
    def setUp(self):
        self.rt=helpers.GroupTests().runtime()
        torch.manual_seed(20261005);self.op=make_operator(self.rt,'joint',level=1,width=4)
        self.op.condition=torch.tensor([[.2,-.2,-.4,.5]])
        self.positions=torch.tensor([[[1,5]],[[1,5]]]);self.op.outer.positions=self.positions
        self.op.outer.enabled=True;self.h=torch.randn(2,8,8)
    def test_zero_exact_in_both_levels(self):
        self.assertTrue(torch.equal(self.op.outer(self.h),self.h))
        self.assertTrue(torch.equal(self.op.inner(self.h,torch.zeros(2,8),modality_positions=self.positions)[0],self.h))
    def test_initial_parameters_match_original(self):
        for arm in ('outer','inner','joint'):
            torch.manual_seed(20261005);a=make_operator(self.rt,arm,level=1,width=4)
            torch.manual_seed(20261005);b=original_operator(self.rt,arm,level=1,width=4)
            aa={k:v for k,v in a.state_dict().items() if '.shared_core.' not in k}
            bb={k:v for k,v in b.state_dict().items() if '.shared_core.' not in k}
            self.assertEqual(set(aa),set(bb))
            self.assertTrue(all(torch.equal(aa[k],bb[k]) for k in aa))
    def test_relative_units_scale_with_hidden_amplitude(self):
        with torch.no_grad():self.op.inner.film.bias.fill_(.2)
        self.op.inner.norm.eps=0
        gamma,beta=self.op.inner.conditioned_affine(self.h,self.positions)
        scaled_gamma,scaled_beta=self.op.inner.conditioned_affine(10*self.h,self.positions)
        torch.testing.assert_close(scaled_gamma,10*gamma,rtol=2e-5,atol=1e-6)
        torch.testing.assert_close(scaled_beta,10*beta,rtol=2e-5,atol=1e-6)
    def test_gradient_and_text_unconditional_preservation(self):
        with torch.no_grad():
            self.op.outer.film.bias.fill_(.2);self.op.inner.film.bias.fill_(.2)
            self.op.outer.film.weight.normal_(std=.01);self.op.inner.film.weight.normal_(std=.01)
        for branch in ('outer','inner'):
            out=self.op.outer(self.h) if branch=='outer' else self.op.inner(self.h,torch.zeros(2,8),modality_positions=self.positions)[0]
            self.assertTrue(torch.equal(out[1],self.h[1]))
            self.assertTrue(torch.equal(out[0,:2],self.h[0,:2]))
            self.assertTrue(torch.equal(out[0,6:],self.h[0,6:]))
            (out-self.h-.1).square().mean().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.op.encoder.parameters() if p.grad is not None),0)
        for core in self.op.shared_core.values():
            self.assertTrue(all(p.grad is None for p in core.parameters()))


if __name__=='__main__':unittest.main()
