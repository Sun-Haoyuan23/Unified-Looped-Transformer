import unittest
from types import SimpleNamespace
import torch
from torch import nn
from runtime import tensor_hash
from two_level_replay import replay_levels,install_levels


class Operator(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.tensor(0.))
        self.condition=torch.zeros(1,4);self.enabled=False;self.shared_core=nn.ModuleList([nn.Identity(),nn.Identity()])
    def set_time(self,value):self.condition[:,3]=value


class RuntimeFixture:
    def __init__(self):
        self.device=torch.device('cpu');self.grid=torch.linspace(0,1,5);self.cfg=5.
        self.original=[nn.Identity() for _ in range(10)]
        self.model=SimpleNamespace(diffusion_head_a=nn.ModuleList(self.original),
            showo=SimpleNamespace(config=SimpleNamespace(use_cache=True),model=SimpleNamespace(layers=nn.ModuleList([nn.Identity() for _ in range(28)]))),
            und_trans=SimpleNamespace(layers=[None]*26))
        self.outer=Operator();self.inner=Operator();self.calls=0
    def inputs(self,prompt):return dict(modality_positions=torch.tensor([[[0,2]]]))
    def from_prefix(self,*args):raise AssertionError('Old prefix must never be reused')
    def velocity(self,x,t,options):
        self.calls+=1;v=.125*x+.1*t
        if self.outer.enabled:v=v+self.outer.weight*(self.outer.condition[0,3]+.05)*x
        if any(op is self.inner for op in self.model.diffusion_head_a):
            v=v+self.inner.weight*(.5-self.inner.condition[0,3])*x
        return v


class TwoLevelTests(unittest.TestCase):
    def setUp(self):
        self.rt=RuntimeFixture();x=torch.arange(8,dtype=torch.float32).reshape(2,1,2,2)/8
        self.state=dict(latent=x,state_fingerprint=tensor_hash(x),time_grid=self.rt.grid.clone(),CFG=5.,prompt='train',flow_index=0)

    def test_zero_exact_and_full_velocity_calls(self):
        baseline,_=replay_levels(self.rt,self.state,window=0)
        result,cost=replay_levels(self.rt,self.state,self.rt.outer,self.rt.inner,window=2)
        self.assertTrue(torch.equal(result,baseline))
        self.assertEqual(cost['network_velocity_calls'],4)
        self.assertEqual(cost['outer_shared_revisit_calls'],8)
        self.assertEqual(cost['inner_shared_revisit_calls'],4)
        self.assertEqual(self.rt.calls,8)

    def test_checkpoint_preserves_bound_times_and_both_gradients(self):
        gradients=[];values=[]
        for checkpoint in (False,True):
            self.rt.outer.weight.data.fill_(.02);self.rt.inner.weight.data.fill_(.03)
            self.rt.outer.weight.grad=None;self.rt.inner.weight.grad=None
            final,_=replay_levels(self.rt,self.state,self.rt.outer,self.rt.inner,window=2,checkpoint_rollout=checkpoint)
            final.square().mean().backward();values.append(final.detach())
            gradients.append(torch.stack([self.rt.outer.weight.grad,self.rt.inner.weight.grad]))
        self.assertTrue(torch.equal(values[0],values[1]))
        self.assertTrue(torch.equal(gradients[0],gradients[1]))
        self.assertTrue(bool((gradients[0]!=0).all()))

    def test_context_restores_on_exception(self):
        with self.assertRaises(RuntimeError):
            with install_levels(self.rt):
                self.assertFalse(self.rt.model.showo.config.use_cache)
                self.rt.model.diffusion_head_a=nn.ModuleList([self.rt.inner])
                raise RuntimeError('test exit')
        self.assertTrue(self.rt.model.showo.config.use_cache)
        self.assertEqual(len(self.rt.model.diffusion_head_a),10)


if __name__=='__main__':unittest.main()
