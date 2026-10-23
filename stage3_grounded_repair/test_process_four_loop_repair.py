import unittest
import torch
from process_four_loop_repair import normalized_hidden,hidden_target,hidden_loss,process_loss
from multibudget_response_repair import shared_response_prefixes
from test_anchored_recurrent_repair import Branch

class Checks(unittest.TestCase):
    def test_target_detach_and_fixed_scale(self):
        s=torch.randn(1,4,5,requires_grad=True);t=torch.randn_like(s,requires_grad=True)
        y,scale=hidden_target(s,t,False)
        self.assertFalse(y.requires_grad or scale.requires_grad)
        pred=torch.randn_like(s,requires_grad=True)
        hidden_loss(dict(outer=pred,inner=pred),dict(outer=y,inner=y),dict(outer=scale,inner=scale)).backward()
        self.assertIsNone(s.grad);self.assertIsNone(t.grad);self.assertGreater(float(pred.grad.abs().sum()),0)
    def test_zero_feedback_process_gradient_connected_anchor(self):
        torch.manual_seed(77);b=Branch(torch.float32);h=torch.randn(2,5,3);u=torch.zeros_like(h,requires_grad=True)
        out,_=shared_response_prefixes(b,h,u,{},(4,),True)
        self.assertTrue(torch.equal(out[4],h))
        (normalized_hidden(out[4])-normalized_hidden(torch.randn_like(h))).square().mean().backward()
        self.assertGreater(float(u.grad.abs().sum()),0)
        self.assertTrue(all(float(p.grad.abs().max())<1e-5 for p in b.parameters()))
    def test_preservation_zero_and_equal_branch_weight(self):
        s=torch.randn(1,4,5);y,scale=hidden_target(s,s,True)
        targets=dict(outer=y,inner=y);scales=dict(outer=scale,inner=scale)
        self.assertEqual(float(hidden_loss(dict(outer=s,inner=s),targets,scales)),0)
        p=s+torch.randn_like(s)*.1;single=(normalized_hidden(p)-y).square().mean()/scale
        self.assertTrue(torch.equal(hidden_loss(dict(outer=p,inner=s),targets,scales),single*.5))
    def test_total_objective_weights(self):
        d=torch.zeros(2,1,2,2,requires_grad=True);gold=torch.ones_like(d);final=d*2
        target=dict(delta=gold,final=gold,preservation=False);h=torch.randn(1,4,3,requires_grad=True)
        y,scale=hidden_target(h.detach(),h.detach()+1,False)
        total,p,t,q=process_loss(d,target,gold,final,torch.zeros_like(d),dict(outer=h,inner=h),dict(outer=y,inner=y),dict(outer=scale,inner=scale))
        self.assertTrue(torch.equal(total,.5*(p+t)+q));total.backward();self.assertIsNotNone(d.grad)
if __name__=='__main__':unittest.main()
