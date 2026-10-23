import unittest
import torch
from torch import nn
from routed_shared_revisit import route_attention_mask,RoutedSharedRevisit,find_prompt_token_spans


class Core(nn.Module):
    def __init__(self):super().__init__();self.scale=nn.Parameter(torch.ones(()))
    def forward(self,hidden_states,attention_mask,**kwargs):
        return (hidden_states+attention_mask.sum(-1).squeeze(1)[...,None]*self.scale,)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.mask=torch.zeros(2,1,8,8);self.positions=torch.tensor([[[3,5]],[[3,5]]]);self.spans={'subject':[1],'reference':[2]}

    def test_only_conditional_image_queries_object_keys_change(self):
        routed=route_attention_mask(self.mask,self.positions,self.spans,'left of',2,2)
        self.assertTrue(torch.equal(routed[1],self.mask[1]))
        self.assertTrue(torch.equal(routed[0,:,:4],self.mask[0,:,:4]))
        self.assertTrue(torch.equal(routed[0,:,:,3:],self.mask[0,:,:,3:]))
        self.assertEqual(int((routed!=self.mask).sum()),8)
        self.assertLess(float(routed[0,0,5,1]),float(routed[0,0,4,1]))

    def test_opposite_swaps_objects_and_zero_is_exact(self):
        forward=route_attention_mask(self.mask,self.positions,self.spans,'below',2,2)
        opposite=route_attention_mask(self.mask,self.positions,self.spans,'below',2,2,opposite=True)
        self.assertTrue(torch.equal(forward[0,0,4:,1],opposite[0,0,4:,2]))
        self.assertTrue(torch.equal(route_attention_mask(self.mask,self.positions,self.spans,'below',2,2,strength=0),self.mask))

    def test_blocked_object_keys_rejected(self):
        self.mask[0,0,4:,1]=-1000
        with self.assertRaisesRegex(ValueError,'causally allowed'):route_attention_mask(self.mask,self.positions,self.spans,'above',2,2)

    def test_zero_identity_and_output_scope_shared_frozen(self):
        core=Core();op=RoutedSharedRevisit(core,self.spans,'left of',2,2,strength=0)
        h=torch.randn(2,8,3);options={'attention_mask':self.mask,'modality_positions':self.positions}
        self.assertTrue(torch.equal(op(h,None,**options)[0],h))
        op.strength=1;result=op(h,None,**options)[0]
        self.assertTrue(torch.equal(result[1],h[1]));self.assertTrue(torch.equal(result[0,:4],h[0,:4]))
        self.assertFalse(torch.equal(result[0,4:],h[0,4:]));self.assertIs(op.shared_core,core)
        self.assertFalse(core.scale.requires_grad)

    def test_context_tokenization_locates_multiword_objects(self):
        prompt='a photo of a sports ball left of a traffic light'
        # Character offsets stand in for a tokenizer; exact prompt subsequence
        # location and span boundaries are the properties under test.
        def tokenizer(text,**kwargs):return {'input_ids':[ord(c) for c in text],'offset_mapping':[(i,i+1) for i in range(len(text))]}
        tokens=torch.tensor([[7]+[ord(c) for c in prompt]+[8]])
        ev={'subject':'sports ball','relation':'left of','reference':'traffic light'}
        spans=find_prompt_token_spans(tokenizer,prompt,ev,tokens,len(prompt)+1)
        self.assertEqual(''.join(chr(int(tokens[0,i])) for i in spans['subject']),'sports ball')
        self.assertEqual(''.join(chr(int(tokens[0,i])) for i in spans['reference']),'traffic light')


if __name__=='__main__':unittest.main()
