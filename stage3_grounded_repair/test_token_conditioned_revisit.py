import unittest
import torch
from torch import nn
from token_conditioned_revisit import TokenConditionedRevisit


class Core(nn.Module):
    def __init__(self):
        super().__init__();self.scale=nn.Parameter(torch.tensor(.7))
    def forward(self,hidden_states,adaln_input,**kwargs):
        return (hidden_states+self.scale*torch.sin(hidden_states),)


class TokenRevisitTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261005)
        self.op=TokenConditionedRevisit(Core(),hidden_size=8,level=1,width=4)
        self.op.condition=torch.tensor([[.2,-.2,-.4,.5]])
        self.h=torch.randn(2,8,8)
        self.pos=torch.tensor([[[1,5]],[[1,5]]])

    def call(self):
        return self.op(self.h,torch.zeros(2,8),modality_positions=self.pos)[0]

    def test_zero_exact_and_shared_core_frozen(self):
        self.assertTrue(torch.equal(self.call(),self.h))
        self.assertFalse(self.op.shared_core.scale.requires_grad)
        self.assertEqual(self.op.executed_core_calls,2)

    def test_fields_are_spatial_and_evidence_dependent(self):
        with torch.no_grad():self.op.film.weight.fill_(.1)
        _,beta=self.op.conditioned_affine(self.h,self.pos)
        self.assertGreater(float((beta[:,2]-beta[:,5]).abs().sum()),0.)
        prior=beta.clone();self.op.condition[:,0]+=1
        _,changed=self.op.conditioned_affine(self.h,self.pos)
        self.assertGreater(float((changed-prior).abs().sum()),0.)
        output=self.call()
        self.assertTrue(torch.equal(output[1],self.h[1]))
        self.assertTrue(torch.equal(output[0,:2],self.h[0,:2]))
        self.assertTrue(torch.equal(output[0,6:],self.h[0,6:]))

    def test_gradient_reaches_zero_initialized_output_not_base(self):
        target=self.h.clone();target[0,2:6]+=.1
        (self.call()-target).square().mean().backward()
        self.assertGreater(float(self.op.film.weight.grad.abs().sum()),0.)
        self.assertIsNone(self.op.shared_core.scale.grad)

    def test_reject_non_square_visual_tokens(self):
        with self.assertRaises(ValueError):
            self.op.conditioned_affine(self.h,torch.tensor([[[1,4]],[[1,4]]]))


if __name__=='__main__':unittest.main()
