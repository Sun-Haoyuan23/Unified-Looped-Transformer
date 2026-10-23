import unittest
import torch
from multibudget_response_repair import shared_response_prefixes
from process_alignment_diagnostics import recomputed_reference

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.shared_core=torch.nn.ModuleList([torch.nn.Linear(3,3,bias=False,dtype=torch.float64) for _ in range(2)]);self.calls=0
    def run_core(self,i,h,kw):self.calls+=1;return torch.tanh(self.shared_core[i](h))

class ReferenceTests(unittest.TestCase):
    def test_nonzero_feedback_forward_and_fp64_derivatives_match(self):
        torch.manual_seed(80);b=Toy();h=torch.randn(2,4,3,dtype=torch.float64);fb=torch.randn_like(h)*.02
        a=shared_response_prefixes(b,h,fb,{},(4,),True)[0][4];ga=torch.autograd.grad(a.square().mean(),tuple(b.parameters()))
        c=recomputed_reference(b,h,fb,{},4);gc=torch.autograd.grad(c.square().mean(),tuple(b.parameters()))
        self.assertTrue(torch.equal(a,c))
        for x,y in zip(ga,gc):self.assertTrue(torch.allclose(x,y,rtol=1e-10,atol=1e-12))
    def test_actual_calls_and_parameter_identity(self):
        b=Toy();h=torch.zeros(2,4,3,dtype=torch.float64);ids=[id(p) for p in b.parameters()]
        shared_response_prefixes(b,h,h,{},(4,),True);self.assertEqual(b.calls,10)
        b.calls=0;recomputed_reference(b,h,h,{},4);self.assertEqual(b.calls,16);self.assertEqual(ids,[id(p) for p in b.parameters()])

if __name__=='__main__':unittest.main()
