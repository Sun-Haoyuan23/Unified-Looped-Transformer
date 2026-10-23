import unittest
import torch
from torch import nn
from object_transport import target_box,transport_field,ObjectTransportRevisit


class Core(nn.Module):
    def __init__(self):super().__init__();self.weight=nn.Parameter(torch.ones(()))
    def forward(self,hidden_states,**kwargs):return (hidden_states*self.weight*1.1,)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.boxes={'subject':[100,100,230,250,.9],'reference':[180,150,350,330,.9]}
        self.masks={'subject':torch.zeros(432,432),'reference':torch.zeros(432,432)}
        self.masks['subject'][100:250,100:230]=1;self.masks['reference'][150:330,180:350]=1

    def test_target_direction_and_size(self):
        for relation,axis,low in [('left of',0,True),('right of',0,False),('above',1,True),('below',1,False)]:
            subject,_=target_box(self.boxes['subject'],relation,'subject')
            reference,_=target_box(self.boxes['reference'],relation,'reference')
            ca=(subject[axis]+subject[axis+2])/2;cb=(reference[axis]+reference[axis+2])/2
            self.assertEqual(ca<cb,low)
            opposite,_=target_box(self.boxes['subject'],relation,'subject',opposite=True)
            self.assertAlmostEqual((opposite[axis]+opposite[axis+2])/2,.75 if low else .25)

    def test_zero_transport_dtype_shape_finite(self):
        x=torch.randn(1,3,12,12,dtype=torch.bfloat16)
        self.assertTrue(torch.equal(transport_field(x,self.masks,self.boxes,'above',strength=0),x))
        moved=transport_field(x,self.masks,self.boxes,'above')
        self.assertEqual(moved.dtype,x.dtype);self.assertEqual(moved.shape,x.shape)
        self.assertTrue(bool(torch.isfinite(moved).all()));self.assertFalse(torch.equal(moved,x))

    def test_shared_zero_and_branch_scope(self):
        core=Core();op=ObjectTransportRevisit(core,self.masks,self.boxes,'left of',2,2,strength=0)
        x=torch.randn(2,8,3);kwargs={'modality_positions':torch.tensor([[[3,5]],[[3,5]]])}
        self.assertTrue(torch.equal(op(x,None,**kwargs)[0],x))
        op.strength=1;out=op(x,None,**kwargs)[0]
        self.assertTrue(torch.equal(out[1],x[1]));self.assertTrue(torch.equal(out[0,:4],x[0,:4]))
        self.assertIs(op.shared_core,core);self.assertFalse(core.weight.requires_grad)


if __name__=='__main__':unittest.main()
