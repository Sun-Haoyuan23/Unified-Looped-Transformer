import unittest
import torch
from process_alignment_diagnostics import direction_stats,recomputed_reference
from multibudget_response_repair import shared_response_prefixes


class ToyBranch(torch.nn.Module):
    def __init__(self):
        super().__init__();self.shared_core=torch.nn.ModuleList([torch.nn.Linear(3,3,bias=False,dtype=torch.float64)])
    def run_core(self,index,h,kwargs):return torch.tanh(self.shared_core[index](h))


class DiagnosticsTest(unittest.TestCase):
    def test_connected_references_cancel_at_zero_in_fp64(self):
        torch.manual_seed(79);branch=ToyBranch();h=torch.randn(2,4,3,dtype=torch.float64)
        for budget in (1,2,4):
            for fresh in (False,True):
                branch.zero_grad(set_to_none=True);feedback=torch.zeros_like(h,requires_grad=True)
                out=recomputed_reference(branch,h,feedback,{},budget) if fresh else shared_response_prefixes(branch,h,feedback,{},(budget,),True)[0][budget]
                self.assertTrue(torch.equal(out,h));out.square().mean().backward()
                self.assertLess(float(branch.shared_core[0].weight.grad.abs().max()),1e-12)
                self.assertGreater(float(feedback.grad.abs().sum()),0.)

    def test_direction_metrics_do_not_invent_zero_target_cosines(self):
        a=torch.tensor([1.,2.]);q=direction_stats(a,a)
        self.assertAlmostEqual(q['cosine'],1.);self.assertEqual(q['mse'],0.)
        z=direction_stats(a,torch.zeros_like(a));self.assertIsNone(z['cosine']);self.assertIsNone(z['relative_target_mse'])


if __name__=='__main__':unittest.main()
