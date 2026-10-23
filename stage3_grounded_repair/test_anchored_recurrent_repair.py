import unittest
import torch
from torch import nn
from anchored_recurrent_repair import revisit_hidden

class Branch(nn.Module):
    def __init__(self,dtype=torch.float64):
        super().__init__();self.shared_core=nn.ModuleList([nn.Linear(3,3,dtype=dtype),nn.Linear(3,3,dtype=dtype)])
    def run_core(self,i,h,kw):return torch.tanh(self.shared_core[i](h))

class Checks(unittest.TestCase):
    def test_zero_feedback_identity_and_connected_reference_gradient(self):
        torch.manual_seed(75);b=Branch();h=torch.randn(2,5,3,dtype=torch.float64);u=torch.zeros_like(h,requires_grad=True)
        # Exact identity must hold even for nonzero trainable core weights.
        hs,calls=revisit_hidden(b,h,u,{},(1,2),True)
        for x in hs.values():self.assertTrue(torch.equal(x,h))
        hs[2].sum().backward();self.assertGreater(float(u.grad.abs().sum()),0.)
        self.assertTrue(all(float(p.grad.abs().max())<1e-12 for p in b.parameters()))
        self.assertEqual(calls,6)
    def test_bfloat16_identity(self):
        torch.manual_seed(75);b=Branch(torch.bfloat16);h=torch.randn(2,5,3,dtype=torch.bfloat16)
        hs,_=revisit_hidden(b,h,torch.zeros_like(h),{},(1,2),True)
        self.assertTrue(all(torch.equal(q,h) for q in hs.values()))
    def test_nested_budget_and_shared_parameter_reuse(self):
        torch.manual_seed(75);b=Branch(torch.float32);h=torch.randn(2,5,3);u=torch.randn_like(h)*.1
        together,c=revisit_hidden(b,h,u,{},(1,2),True)
        one,c1=revisit_hidden(b,h,u,{},(1,),True);two,c2=revisit_hidden(b,h,u,{},(2,),True)
        self.assertTrue(torch.equal(together[1],one[1]));self.assertTrue(torch.equal(together[2],two[2]))
        self.assertEqual((c,c1,c2),(6,4,6));self.assertEqual(len(list(b.parameters())),4)
    def test_reject_recurrent_kv(self):
        b=Branch();h=torch.zeros(2,5,3,dtype=torch.float64)
        for kw in ({'use_cache':True},{'past_key_value':object()}):
            with self.assertRaises(ValueError):revisit_hidden(b,h,h,kw)

if __name__=='__main__':unittest.main()
