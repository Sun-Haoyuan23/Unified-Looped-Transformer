import unittest
from types import SimpleNamespace
import torch
from torch import nn
from process_supervision import recurrent_trace, frozen_feature_normalizers, process_terms, visual_span


class Core(nn.Module):
    def forward(self,x):return (x+1,)


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.positions=torch.tensor([[[0,5]],[[0,5]]])
        self.original=torch.zeros(2,1,2,2)
        self.target=torch.ones_like(self.original)
        self.features={'outer':torch.randn(2,7,6)}

    def test_hooks_removed_after_exception(self):
        core=Core();op=SimpleNamespace(outer=SimpleNamespace(shared_core=[core]),inner=None)
        with self.assertRaisesRegex(RuntimeError,'interrupted'):
            with recurrent_trace(op):raise RuntimeError('interrupted')
        self.assertEqual(len(core._forward_hooks),0)

    def test_nested_visits_and_missing_prefix(self):
        core=Core();op=SimpleNamespace(outer=SimpleNamespace(shared_core=[core]),inner=None)
        with recurrent_trace(op) as trace:
            first=core(torch.zeros(2,7,6))[0];second=core(first)[0]
        self.assertTrue(torch.equal(trace['outer'][0],first))
        self.assertTrue(torch.equal(trace['outer'][1],second))
        with self.assertRaisesRegex(ValueError,'Missing'):
            with recurrent_trace(op):core(first)
        self.assertEqual(len(core._forward_hooks),0)

    def test_feature_target_detached_and_gradient_enters_core_states(self):
        target=self.features['outer'].requires_grad_()
        traces={'outer':[torch.randn(2,7,6,requires_grad=True) for _ in range(2)]}
        scales=frozen_feature_normalizers(traces,{'outer':target},self.positions)
        terms=process_terms({1:self.target,2:self.target},traces,self.original,self.target,
            {'outer':target},self.positions,False,scales)
        self.assertAlmostEqual(float(terms['state']),0.)
        self.assertAlmostEqual(float(terms['hidden']),1.,places=5)
        terms['hidden'].backward()
        self.assertIsNone(target.grad)
        self.assertTrue(all(float(q.grad.abs().sum())>0 for q in traces['outer']))

    def test_nonvisual_and_unconditional_changes_do_not_change_feature_loss(self):
        traces={'outer':[self.features['outer'].clone() for _ in range(2)]}
        scales={'outer':[torch.tensor(1.),torch.tensor(1.)]}
        for q in traces['outer']:
            q[1].add_(1000);q[0,0].add_(1000);q[0,5:].add_(1000)
        parts=process_terms({1:self.target,2:self.target},traces,self.original,self.target,
            self.features,self.positions,False,scales)
        self.assertEqual(float(parts['hidden']),0.)

    def test_invalid_span_and_divergent_CFG_rejected(self):
        with self.assertRaises(ValueError):visual_span(self.features['outer'],torch.tensor([[[0,6]],[[0,6]]]))
        delta=self.target.clone();delta[1].zero_()
        traces={'outer':[self.features['outer']]*2}
        with self.assertRaisesRegex(ValueError,'identical'):
            process_terms({1:delta,2:delta},traces,self.original,self.target,self.features,
                self.positions,False,{'outer':[torch.tensor(1.),torch.tensor(1.)]})


if __name__=='__main__':unittest.main()
