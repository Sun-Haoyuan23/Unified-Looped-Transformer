import unittest
import torch


def state_weight(preservation):return .5/(35 if preservation else 6)


class FullBatchTests(unittest.TestCase):
    def test_group_weights_and_gradient_accumulation(self):
        self.assertAlmostEqual(6*state_weight(False),.5)
        self.assertAlmostEqual(35*state_weight(True),.5)
        p=torch.nn.Parameter(torch.tensor([.3,-.4],dtype=torch.float64))
        cases=[(torch.tensor([i/10.,1.],dtype=torch.float64),i/20.,i>=6) for i in range(41)]
        dense=sum(state_weight(keep)*(p@x-y).square() for x,y,keep in cases)
        expected=torch.autograd.grad(dense,p)[0]
        for x,y,keep in cases:(state_weight(keep)*(p@x-y).square()).backward()
        torch.testing.assert_close(p.grad,expected,rtol=1e-13,atol=1e-13)

    def test_teacher_target_is_detached(self):
        p=torch.nn.Parameter(torch.tensor([.3,-.4],dtype=torch.float64));target=torch.ones(2,dtype=torch.float64,requires_grad=True)
        loss=(p-target.detach()).square().mean();loss.backward()
        self.assertIsNone(target.grad);self.assertIsNotNone(p.grad)


if __name__=='__main__':unittest.main()
