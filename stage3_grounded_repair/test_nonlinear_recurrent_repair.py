import unittest
from collections import Counter
import torch
from torch import nn
from nonlinear_recurrent_repair import NonlinearRecurrentRepair,balanced_schedule,normalized_losses


class Branch(nn.Module):
    def __init__(self,encoder):
        super().__init__();self.shared_core=nn.ModuleList([nn.Linear(2,2)])
        self.low_rank=nn.ModuleDict({'adapter':nn.Linear(2,2)})
        self.encoder=encoder;self.readout=nn.Linear(2,2);self.feedback=nn.Linear(2,2)
        self.learn_core=True


class Operator(nn.Module):
    def __init__(self):
        super().__init__();encoder=nn.Linear(2,2)
        self.outer,self.inner=Branch(encoder),Branch(encoder)


class Tests(unittest.TestCase):
    def test_freeze_preserves_loaded_core_route(self):
        for arm in ('no_loop','frozen_loop','trainable_loop'):
            op=Operator();before={k:v.clone() for k,v in op.state_dict().items()}
            repair=NonlinearRecurrentRepair(op,nn.Linear(2,2),arm)
            self.assertTrue(op.outer.learn_core and op.inner.learn_core)
            for name,p in op.named_parameters():
                self.assertTrue(torch.equal(p,before[name]))
                if '.shared_core.' in name or '.readout.' in name:self.assertFalse(p.requires_grad)
                elif 'low_rank' in name:self.assertEqual(p.requires_grad,arm=='trainable_loop')
                else:self.assertTrue(p.requires_grad)
            params=repair.trainable_parameters();self.assertEqual(len(params),len({id(p) for p in params}))

    def test_fixed_full_preservation_coverage(self):
        teachers=['t0','t1','t2'];correct=[f'c{i}' for i in range(35)]
        schedule=balanced_schedule(teachers,correct)
        self.assertEqual(schedule,balanced_schedule(teachers,correct));self.assertEqual(len(schedule),210)
        counts=Counter(sid for sid,preserve in schedule)
        self.assertTrue(all(counts[sid]==35 for sid in teachers))
        self.assertTrue(all(counts[sid]==3 for sid in correct))
        self.assertTrue(all(flag==(i%2==1) for i,(sid,flag) in enumerate(schedule)))

    def test_train_targets_detached_from_loss(self):
        delta=torch.ones(2,requires_grad=True);gold=torch.full((2,),2.,requires_grad=True)
        target=torch.full((2,),3.,requires_grad=True);final=delta*2
        loss,physical,terminal=normalized_losses(delta,gold,torch.ones(2),final,target,torch.zeros(2),False)
        loss.backward();self.assertIsNotNone(delta.grad)
        self.assertIsNone(gold.grad);self.assertIsNone(target.grad)
        self.assertAlmostEqual(float(physical.detach()),.25)
        self.assertAlmostEqual(float(terminal.detach()),1/9,places=6)

    def test_foreign_cached_state_rejected(self):
        repair=NonlinearRecurrentRepair(Operator(),nn.Linear(2,2),'trainable_loop')
        state=dict(state_fingerprint='x',prompt='p',flow_index=35)
        with self.assertRaisesRegex(ValueError,'different physical state'):
            repair.latent_delta(None,state,dict(_provenance=('y','p',35)),None)


if __name__=='__main__':unittest.main()
