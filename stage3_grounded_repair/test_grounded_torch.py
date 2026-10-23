import unittest
from types import SimpleNamespace
import torch
from grounded_actions import spatial_field,regional_velocity
from evidence_conditioned_revisit import ConditionedRevisit


class GroundedTorchTests(unittest.TestCase):
    def test_fields_complement_and_reverse_with_correct_axis(self):
        for positive, opposite in [('left of','right of'),('above','below')]:
            a,b=spatial_field(positive,7,9,'cpu')
            c,d=spatial_field(opposite,7,9,'cpu')
            torch.testing.assert_close(a+b,torch.ones_like(a),rtol=0,atol=0)
            torch.testing.assert_close(a,c.flip(-1 if positive=='left of' else -2))
            self.assertEqual(a.shape,(1,1,7,9))
            self.assertTrue(bool((a>0).all() and (a<1).all()))
            if positive=='left of':self.assertGreater(float(a[0,0,3,0]),float(a[0,0,3,-1]))
            else:self.assertGreater(float(a[0,0,0,4]),float(a[0,0,-1,4]))

    def test_regional_teacher_branches_and_counters_are_correct(self):
        class Runtime:
            def inputs(self,prompt):return prompt
            def velocity(self,x,t,prompt):return torch.ones_like(x)*(1 if 'cat' in prompt else -1)
        x=torch.zeros(2,3,7,9)
        e={'subject':'cat','reference':'dog','relation':'left of'}
        v,c,b=regional_velocity(Runtime(),x,.3,x,e)
        wrong,_,_=regional_velocity(Runtime(),x,.3,x,e,opposite=True)
        self.assertEqual((c,b),(2,20))
        torch.testing.assert_close(v[0],v[1],rtol=0,atol=0)
        torch.testing.assert_close(v,-wrong,rtol=1e-6,atol=1e-7)
        self.assertGreater(float(v[0,0,3,0]),0.)
        self.assertLess(float(v[0,0,3,-1]),0.)
        zero,_,_=regional_velocity(Runtime(),x,.3,x,e,strength=0.)
        self.assertTrue(torch.equal(zero,x))

    def test_token_probe_updates_only_conditional_image_tokens(self):
        class Core(torch.nn.Module):
            def forward(self,hidden_states,adaln_input,**kwargs):
                return (hidden_states+.2*hidden_states.tanh(),)
        op=ConditionedRevisit(Core(),hidden_size=8,level=1,width=16)
        op.mode='probe';op.probe_beta=torch.randn(5,8)
        h=torch.randn(2,5,8)
        out=op(h,torch.tensor(.3),modality_positions=torch.tensor([[[1,4]],[[1,4]]]))[0]
        self.assertTrue(torch.equal(out[1],h[1]))
        self.assertTrue(torch.equal(out[0,:2],h[0,:2]))
        self.assertFalse(torch.equal(out[0,2:],h[0,2:]))


if __name__=='__main__':unittest.main()
