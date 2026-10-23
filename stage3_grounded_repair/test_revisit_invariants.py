"""Critical tests: exact zero, frozen first core, nonzero evidence gradient."""
import unittest
import torch
from torch import nn
from evidence_conditioned_revisit import ConditionedRevisit


class Core(nn.Module):
    def __init__(self):
        super().__init__();self.proj=nn.Linear(8,8)
    def forward(self,hidden_states,adaln_input,**kwargs):
        return (hidden_states+self.proj(hidden_states).tanh(),)


class Invariants(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.core=Core()
        self.op=ConditionedRevisit(self.core,hidden_size=8,level=1,width=5)
        self.op.condition=torch.randn(1,4)
        self.x=torch.randn(2,6,8)
        self.kw={'modality_positions':torch.tensor([[[1,4]],[[1,4]]])}
    def test_zero_exact(self):
        y=self.op(self.x,None,**self.kw)[0]
        self.assertTrue(torch.equal(y,self.x))
    def test_evidence_receives_gradient(self):
        y=self.op(self.x,None,**self.kw)[0]
        (y*torch.randn_like(y)).sum().backward()
        self.assertGreater(self.op.film.weight.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None for p in self.core.parameters()))
        self.assertTrue(all(not p.requires_grad for p in self.core.parameters()))
    def test_only_conditional_visual_tokens_change(self):
        with torch.no_grad():self.op.film.weight.normal_(std=.2)
        y=self.op(self.x,None,**self.kw)[0]
        self.assertTrue(torch.equal(y[1],self.x[1]))
        self.assertTrue(torch.equal(y[0,:2],self.x[0,:2]))
        self.assertTrue(torch.equal(y[0,5:],self.x[0,5:]))
        self.assertFalse(torch.equal(y[0,2:5],self.x[0,2:5]))
    def test_shared_parameter_identity(self):
        self.assertIs(self.op.shared_core,self.core)
        ids={id(p) for p in self.op.trainable_parameters()}
        self.assertTrue(ids.isdisjoint({id(p) for p in self.core.parameters()}))


if __name__=='__main__': unittest.main()
