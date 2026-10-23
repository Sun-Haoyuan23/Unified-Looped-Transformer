import unittest
from types import SimpleNamespace
import torch
from torch import nn
from two_level_operator import make_operator

class Core(nn.Module):
    def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor(.2))
    def forward(self,hidden_states,**kwargs):return (hidden_states+self.scale*torch.sin(hidden_states),)

class GroupTests(unittest.TestCase):
    def runtime(self):
        return SimpleNamespace(device=torch.device('cpu'),original=[Core() for _ in range(10)],
            model=SimpleNamespace(showo=SimpleNamespace(config=SimpleNamespace(hidden_size=8),
                model=SimpleNamespace(layers=nn.ModuleList([Core() for _ in range(28)]))),
                diffusion_head_config=SimpleNamespace(hidden_size=8)))
    def test_common_initialization_matches(self):
        states=[]
        for arm in ('outer','inner','joint'):
            torch.manual_seed(20261005);op=make_operator(self.runtime(),arm,width=4)
            states.append(op.encoder.state_dict())
        for key in states[0]:
            self.assertTrue(torch.equal(states[0][key],states[1][key]))
            self.assertTrue(torch.equal(states[0][key],states[2][key]))
    def test_joint_memory_is_shared_and_optimizer_deduplicated(self):
        op=make_operator(self.runtime(),'joint',width=4)
        self.assertIs(op.outer.encoder,op.inner.encoder)
        params=op.trainable_parameters();self.assertEqual(len(params),len({id(p) for p in params}))
        for core in op.shared_core.values():
            for p in core.parameters():self.assertFalse(p.requires_grad)
    def test_checkpoint_excludes_frozen_cores(self):
        op=make_operator(self.runtime(),'joint',width=4)
        state={k:v for k,v in op.state_dict().items() if '.shared_core.' not in k}
        self.assertFalse(any('shared_core' in k for k in state))
        self.assertTrue(any(k.startswith('outer.film') for k in state))
        self.assertTrue(any(k.startswith('inner.film') for k in state))

if __name__=='__main__':unittest.main()
