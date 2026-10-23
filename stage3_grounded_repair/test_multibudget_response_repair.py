import unittest
import torch
from test_anchored_recurrent_repair import Branch
from anchored_recurrent_repair import revisit_hidden
from multibudget_response_repair import shared_response_prefixes

class Checks(unittest.TestCase):
    def test_v75_prefixes_preserved(self):
        torch.manual_seed(76);b=Branch(torch.float32);h=torch.randn(2,5,3);u=torch.randn_like(h)*.1
        old,_=revisit_hidden(b,h,u,{},(1,2),True);new,c=shared_response_prefixes(b,h,u,{},(1,2,4),True)
        self.assertTrue(torch.equal(old[1],new[1]));self.assertTrue(torch.equal(old[2],new[2]));self.assertEqual(c,10)
    def test_four_loop_zero_feedback_and_core_reference_gradient(self):
        torch.manual_seed(76);b=Branch();h=torch.randn(2,5,3,dtype=torch.float64);u=torch.zeros_like(h,requires_grad=True)
        out,c=shared_response_prefixes(b,h,u,{},(1,2,4),True)
        self.assertTrue(all(torch.equal(h,q) for q in out.values()));self.assertEqual(c,10)
        out[4].sum().backward();self.assertGreater(float(u.grad.abs().sum()),0.)
        self.assertTrue(all(float(p.grad.abs().max())<1e-12 for p in b.parameters()))
    def test_no_extra_parameters_and_declared_call_cost(self):
        b=Branch(torch.float32);h=torch.randn(2,5,3);u=torch.randn_like(h)*.1;before=[id(p) for p in b.parameters()]
        _,anchor=shared_response_prefixes(b,h,u,{},(4,),True);_,plain=shared_response_prefixes(b,h,u,{},(4,),False)
        self.assertEqual((anchor,plain),(10,8));self.assertEqual(before,[id(p) for p in b.parameters()])
if __name__=='__main__':unittest.main()
