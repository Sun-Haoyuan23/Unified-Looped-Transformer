"""Instance transport oracle in latent or shared-block feature space.

Segmentation comes from an external OFFLINE teacher. A learned MMU-grounded
localizer would be needed for deployment. No pixels or terminal labels are used.
"""
import torch
from torch import nn
from torch.nn import functional as F


def target_box(box,relation,role,opposite=False):
    if relation not in ('left of','right of','above','below'):raise ValueError('Unknown relation')
    if role not in ('subject','reference'):raise ValueError('Unknown object role')
    x1,y1,x2,y2=[float(v)/432 for v in box[:4]]
    width,height=x2-x1,y2-y1
    if width<=0 or height<=0:raise ValueError('Empty object box')
    horizontal=relation in ('left of','right of')
    first_low=relation in ('left of','above')
    low=first_low if role=='subject' else not first_low
    if opposite:low=not low
    desired=.25 if low else .75
    scale=min(1.,.4/(width if horizontal else height))
    new_width,new_height=width*scale,height*scale
    cx,cy=(x1+x2)/2,(y1+y2)/2
    if horizontal:cx=desired
    else:cy=desired
    cx=max(new_width/2,min(1-new_width/2,cx));cy=max(new_height/2,min(1-new_height/2,cy))
    return [cx-new_width/2,cy-new_height/2,cx+new_width/2,cy+new_height/2],scale


def transport_field(field,masks,boxes,relation,opposite=False,strength=1.):
    if field.ndim!=4 or field.shape[0]!=1:raise ValueError('Transport expects one image field')
    if not 0<=strength<=1:raise ValueError('Transport strength out of range')
    if strength==0:return field.clone()
    height,width=field.shape[-2:]
    source={role:F.interpolate(masks[role].to(field.device).float()[None,None],size=(height,width),mode='area').clamp(0,1)
            for role in ('subject','reference')}
    union=torch.maximum(source['subject'],source['reference'])
    original=field.float()
    # Local context is a fill proxy; the remaining frozen flow must regenerate
    # the vacated region. This is not a guaranteed inpainting operation.
    blur=F.avg_pool2d(original,9,stride=1,padding=4,count_include_pad=False)
    result=original*(1-union)+blur*union
    xs=(torch.arange(width,device=field.device)+.5)/width
    ys=(torch.arange(height,device=field.device)+.5)/height
    yy,xx=torch.meshgrid(ys,xs,indexing='ij')
    for role in ('subject','reference'):
        box=boxes[role];target,scale=target_box(box,relation,role,opposite)
        old_cx=(box[0]+box[2])/864;old_cy=(box[1]+box[3])/864
        new_cx=(target[0]+target[2])/2;new_cy=(target[1]+target[3])/2
        grid=torch.stack((2*((xx-new_cx)/scale+old_cx)-1,2*((yy-new_cy)/scale+old_cy)-1),dim=-1)[None]
        moved=F.grid_sample(original,grid,align_corners=False,padding_mode='zeros')
        moved_mask=F.grid_sample(source[role],grid,align_corners=False,padding_mode='zeros').clamp(0,1)
        result=result*(1-moved_mask)+moved*moved_mask
    return (original+strength*(result-original)).to(field.dtype)


class ObjectTransportRevisit(nn.Module):
    def __init__(self,core,masks,boxes,relation,height,width,strength=1.,opposite=False):
        super().__init__();self.shared_core=core;core.requires_grad_(False)
        self.masks=masks;self.boxes=boxes;self.relation=relation;self.height=height;self.width=width
        self.strength=strength;self.opposite=opposite;self.executed_core_calls=2

    def set_time(self,value):pass

    def forward(self,hidden_states,adaln_input,**kwargs):
        start,length=kwargs['modality_positions'][0,0].tolist()
        if length-1!=self.height*self.width:raise ValueError('Transport feature grid mismatch')
        moved=hidden_states.clone()
        field=hidden_states[0:1,start+1:start+length].transpose(1,2).reshape(1,-1,self.height,self.width)
        moved_field=transport_field(field,self.masks,self.boxes,self.relation,self.opposite,self.strength)
        moved[0:1,start+1:start+length]=moved_field.flatten(2).transpose(1,2)
        anchor=self.shared_core(hidden_states=hidden_states,adaln_input=adaln_input,**kwargs)[0]
        proposal=self.shared_core(hidden_states=moved,adaln_input=adaln_input,**kwargs)[0]
        mask=torch.zeros_like(hidden_states[...,:1]);mask[0,start+1:start+length]=1
        return (hidden_states+(proposal-anchor)*mask,)
