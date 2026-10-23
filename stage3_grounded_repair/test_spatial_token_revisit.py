import unittest
import torch
from torch import nn
from spatial_token_revisit import SpatialTokenRevisit


class Core(nn.Module):
    def __init__(self):
        super().__init__();self.scale=nn.Parameter(torch.tensor(.7))
    def forward(self,hidden_states,adaln_input,**kwargs):
        return (hidden_states+self.scale*torch.sin(hidden_states),)


class SpatialTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261005)
        self.op=SpatialTokenRevisit(Core(),hidden_size=8,level=1,width=4)
        self.op.condition=torch.tensor([[.2,-.2,-.4,.5]])
        self.h=torch.randn(2,13,8)
        self.pos=torch.tensor([[[1,10]],[[1,10]]])
    def call(self):return self.op(self.h,torch.zeros(2,8),modality_positions=self.pos)[0]
    def test_exact_zero_and_frozen_core(self):
        self.assertTrue(torch.equal(self.call(),self.h))
        self.assertFalse(self.op.shared_core.scale.requires_grad)
    def test_transport_gradient_at_zero(self):
        target=self.h.clone();target[0,2:11]=self.h[0,2:11].reshape(3,3,8).flip(1).reshape(9,8)
        (self.call()-target).square().mean().backward()
        self.assertGreater(float(self.op.displacement.weight.grad.abs().sum()),0.)
        self.assertIsNone(self.op.shared_core.scale.grad)
    def test_text_unconditional_and_special_tokens_preserved(self):
        with torch.no_grad():self.op.displacement.bias.copy_(torch.tensor([.2,-.3]))
        y=self.call()
        self.assertTrue(torch.equal(y[1],self.h[1]))
        self.assertTrue(torch.equal(y[0,:2],self.h[0,:2]))
        self.assertTrue(torch.equal(y[0,11:],self.h[0,11:]))
        self.assertGreater(float((y[0,2:11]-self.h[0,2:11]).abs().sum()),0.)
    def test_evidence_controls_transport(self):
        with torch.no_grad():self.op.displacement.weight.fill_(.1)
        y=self.op.conditioned_hidden(self.h,self.pos)
        self.op.condition[:,0]+=1
        z=self.op.conditioned_hidden(self.h,self.pos)
        self.assertGreater(float((y-z).abs().sum()),0.)
    def test_reject_non_square_grid(self):
        with self.assertRaises(ValueError):self.op.token_field(self.h,torch.tensor([[[1,9]],[[1,9]]]))


if __name__=='__main__':unittest.main()
