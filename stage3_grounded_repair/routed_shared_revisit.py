"""Object-token routing inside a second visit to the SAME frozen visual block.

The first ten-block generation path is untouched. Only the extra visit changes
image-query -> named-object-text-key attention weights. Conditional image
outputs receive F(h, routed_mask)-F(h, original_mask); text/time/unconditional
outputs are untouched. Routing strength zero is a strict identity.

This is a development repairability probe, not a trained evidence-conditioned
repairer: object identities and target relation are parsed from the existing
MMU question, and all states are intervened for preservation measurement.
"""
import math
import re
import torch
from torch import nn
from grounded_actions import spatial_field


def find_prompt_token_spans(tokenizer,prompt,evidence,text_tokens,image_offset):
    # Use offsets in the complete prompt; separately tokenizing an object loses
    # leading-space BPE context. Then locate the exact prompt in model inputs.
    encoded=tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True)
    ids=list(encoded['input_ids']);offsets=list(encoded['offset_mapping'])
    prefix=text_tokens[0,:image_offset].tolist()
    starts=[i for i in range(len(prefix)-len(ids)+1) if prefix[i:i+len(ids)]==ids]
    if len(starts)!=1:raise ValueError('Prompt token sequence must occur exactly once before image tokens')
    match=re.fullmatch(r'a photo of (?:a|an) (.+?) (left of|right of|above|below) (?:a|an) (.+)',prompt,re.I)
    if match is None or tuple(match.groups())!=(evidence['subject'],evidence['relation'],evidence['reference']):
        raise ValueError('Prompt parse and MMU evidence disagree')
    output={}
    for role,group in [('subject',1),('reference',3)]:
        a,b=match.span(group)
        chosen=[starts[0]+i for i,(left,right) in enumerate(offsets) if left<b and right>a]
        if not chosen:raise ValueError('Object name has no tokens')
        output[role]=chosen
    if set(output['subject'])&set(output['reference']):raise ValueError('Object token spans overlap')
    return output


def route_attention_mask(mask,positions,token_spans,relation,height,width,strength=1.0,opposite=False):
    if mask.ndim!=4 or mask.shape[0]!=2 or mask.shape[1]!=1 or mask.shape[-1]!=mask.shape[-2]:
        raise ValueError('Expected released two-branch additive attention mask')
    if not math.isfinite(strength) or not 0<=strength<=1:raise ValueError('Routing strength must be in [0,1]')
    start,length=positions[0,0].tolist()
    if length-1!=height*width:raise ValueError('Token count differs from visual grid')
    if strength==0:return mask.clone()
    support,reference=spatial_field(relation,height,width,mask.device)
    if opposite:support,reference=reference,support
    result=mask.clone()
    queries=torch.arange(start+1,start+length,device=mask.device)
    for role,field in [('subject',support),('reference',reference)]:
        keys=token_spans[role]
        if not keys or len(set(keys))!=len(keys) or any(i<0 or i>=start for i in keys):
            raise ValueError('Object keys must be distinct text tokens before image block')
        prior=field.flatten().clamp_min(.01).log()*strength
        for key in keys:
            original=mask[0,0,queries,key]
            if not bool((original==0).all()):raise ValueError('Object token attention was not causally allowed')
            result[0,0,queries,key]=original+prior.to(mask.dtype)
    return result


class RoutedSharedRevisit(nn.Module):
    def __init__(self,shared_core,token_spans,relation,height,width,strength=1.0,opposite=False):
        super().__init__();self.shared_core=shared_core
        self.token_spans=token_spans;self.relation=relation;self.height=height;self.width=width
        self.strength=strength;self.opposite=opposite;self.executed_core_calls=2
        self.changed_mask_elements=0
        self.shared_core.requires_grad_(False)

    def set_time(self,value):pass

    def forward(self,hidden_states,adaln_input,**kwargs):
        original=kwargs['attention_mask'];positions=kwargs['modality_positions']
        routed=route_attention_mask(original,positions,self.token_spans,self.relation,
                                    self.height,self.width,self.strength,self.opposite)
        self.changed_mask_elements=int((routed!=original).sum())
        anchor=self.shared_core(hidden_states=hidden_states,adaln_input=adaln_input,**kwargs)[0]
        options=dict(kwargs,attention_mask=routed)
        proposal=self.shared_core(hidden_states=hidden_states,adaln_input=adaln_input,**options)[0]
        mask=torch.zeros_like(hidden_states[...,:1]);start,length=positions[0,0].tolist()
        mask[0,start+1:start+length]=1
        return (hidden_states+(proposal-anchor)*mask,)
