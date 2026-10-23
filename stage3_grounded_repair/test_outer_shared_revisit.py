import unittest
import torch
from torch import nn
from outer_shared_revisit import OuterSharedRevisit

class Core(nn.Module):
    def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor(.2))
    def forward(self,hidden_states,**kwargs):return (hidden_states+self.scale*torch.sin(hidden_states),)
class OuterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261005)
        self.op=OuterSharedRevisit([Core(),Core()],hidden_size=8,level=1,width=4)
        self.op.condition=torch.tensor([[.2,-.2,-.4,.5]]);self.op.positions=torch.tensor([[[1,5]],[[1,5]]])
        self.op.enabled=True;self.h=torch.randn(2,8,8)
    def test_exact_zero(self):
        self.assertTrue(torch.equal(self.op(self.h),self.h));self.assertEqual(self.op.executed_core_calls,4)
    def test_gradient_not_base(self):
        target=self.h.clone();target[0,2:6]+=.1
        (self.op(self.h)-target).square().mean().backward()
        self.assertGreater(float(self.op.film.weight.grad.abs().sum()),0.)
        for p in self.op.shared_core.parameters():self.assertIsNone(p.grad)
    def test_preserve_text_and_unconditional(self):
        with torch.no_grad():self.op.film.weight.fill_(.1)
        out=self.op(self.h)
        self.assertTrue(torch.equal(out[1],self.h[1]));self.assertTrue(torch.equal(out[0,:2],self.h[0,:2]))
        self.assertTrue(torch.equal(out[0,6:],self.h[0,6:]))
    def test_cache_rejected(self):
        with self.assertRaises(ValueError):self.op(self.h,use_cache=True)
    def test_disabled_does_not_revisit(self):
        self.op.enabled=False;self.assertTrue(torch.equal(self.op(self.h),self.h));self.assertEqual(self.op.executed_core_calls,0)
if __name__=='__main__':unittest.main()
