import unittest
import torch
from readout_subspace_solver import weighted_block,solve_fixed_subspace,patch_target_from_permutation

class Tests(unittest.TestCase):
    def test_group_weights_and_exact_fit(self):
        torch.manual_seed(82);a=torch.randn(20,3,dtype=torch.float64);b=torch.randn(4,2,dtype=torch.float64)
        target=torch.cat((a,torch.ones(20,1,dtype=torch.float64)),1)@b
        x,y=weighted_block(a,target,2.,.5);c,q=solve_fixed_subspace(x,y)
        self.assertTrue(torch.allclose(c,b,atol=1e-12,rtol=1e-12));self.assertEqual(q['numerical_rank'],4)
        residual=x@torch.zeros_like(b)-y
        self.assertAlmostEqual(float(residual.square().sum()),.5*float(target.square().mean())/2.)
        xd=torch.cat((x,x[:,-1:]),1);cd,qd=solve_fixed_subspace(xd,y)
        self.assertEqual(qd['numerical_rank'],4);self.assertLess(qd['weighted_objective'],1e-25)
        with self.assertRaises(ValueError):weighted_block(a,target,0.,.5)
    def test_patch_permutation_roundtrip_and_rejection(self):
        p=torch.tensor([3,0,5,2,1,4]);physical=torch.arange(6.).reshape(2,3)
        patches=patch_target_from_permutation(physical,p,(3,2))
        self.assertTrue(torch.equal(patches.flatten()[p],physical.flatten()))
        with self.assertRaises(ValueError):patch_target_from_permutation(physical,torch.zeros(6,dtype=torch.long),(3,2))
if __name__=='__main__':unittest.main()
