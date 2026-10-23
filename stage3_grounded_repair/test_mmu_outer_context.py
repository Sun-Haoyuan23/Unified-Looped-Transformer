import unittest
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import contextmanager
import torch
from mmu_outer_context import mmu_loop_context
from rms_scaled_revisit import RMSOuterRevisit
import test_outer_shared_revisit as helpers


@contextmanager
def fake_install(runtime,operator):yield


class SpanTests(unittest.TestCase):
    def setUp(self):
        self.runtime=SimpleNamespace(device=torch.device('cpu'))
        self.op=RMSOuterRevisit([helpers.Core(),helpers.Core()],hidden_size=8,level=1,width=4)
        self.op.condition=torch.tensor([[.2,-.2,-.4,1.]])
        self.span=dict(visual_start=3,visual_count=4,sequence_length=10)
        self.h=torch.randn(1,10,8)
    def test_zero_exact_single_mmu_branch(self):
        with patch('mmu_outer_context.install',fake_install),mmu_loop_context(self.runtime,self.op,self.span):
            self.assertTrue(torch.equal(self.op(self.h),self.h))
            self.assertEqual(self.op.executed_core_calls,4)
    def test_true_visual_span_preserves_time_and_eoi(self):
        with torch.no_grad():self.op.film.bias.fill_(.2)
        with patch('mmu_outer_context.install',fake_install),mmu_loop_context(self.runtime,self.op,self.span):
            result=self.op(self.h)
            self.assertTrue(torch.equal(result[:,:3],self.h[:,:3]))
            self.assertTrue(torch.equal(result[:,7:],self.h[:,7:]))
            self.assertFalse(torch.equal(result[:,3:7],self.h[:,3:7]))
    def test_restores_context_and_rejects_invalid_span(self):
        old=torch.tensor([[[1,5]]]);self.op.positions=old
        with patch('mmu_outer_context.install',fake_install):
            with self.assertRaises(RuntimeError):
                with mmu_loop_context(self.runtime,self.op,self.span):raise RuntimeError('deliberate')
        self.assertIs(self.op.positions,old);self.assertFalse(self.op.enabled)
        with self.assertRaises(ValueError):
            with mmu_loop_context(self.runtime,self.op,dict(visual_start=3,visual_count=5,sequence_length=10)):pass


if __name__=='__main__':unittest.main()
