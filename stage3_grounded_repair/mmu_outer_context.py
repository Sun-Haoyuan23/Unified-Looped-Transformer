"""Explicit MMU visual span for existing outer shared recurrence.

Keep the official MMU attention mask unchanged. The adapter's visual-only mask
must follow the actual embedding sequence, not the T2I modality-offset convention.
"""
from contextlib import contextmanager
import math
import torch
from outer_shared_revisit import install


def question_inputs(reader,visual,question):
    r=reader.runtime;tok=r.tokenizer
    system=tok('system\nYou are a helpful assistant.<|im_end|>',add_special_tokens=False).input_ids
    user=tok('\n<|im_start|>user\n',add_special_tokens=False).input_ids
    assistant=tok('\n<|im_start|>assistant\n',add_special_tokens=False).input_ids
    query=tok(question,add_special_tokens=False).input_ids
    a=torch.tensor([r.ids['bos_id']]+system+user,device=r.device)[None]
    b=torch.tensor([r.ids['boi_id'],r.ids['eoi_id']]+query+assistant,device=r.device)[None]
    ae=r.model.showo.model.embed_tokens(a);be=r.model.showo.model.embed_tokens(b)
    time=r.model.time_embed(torch.tensor([[1.]],device=r.device),ae.dtype)
    if hasattr(r.model,'time_embed_proj'):time=r.model.time_embed_proj(time)
    inputs=torch.cat((ae,be[:,:1],time,visual,be[:,1:]),dim=1)
    pos=torch.tensor([a.shape[1]+2,r.nmmu],device=r.device)[None,None]
    mask=r.mask_fn(B=1,LEN=inputs.shape[1],modalities=pos,device=r.device,inverted=True).to(inputs.dtype)
    return dict(inputs_embeds=inputs,attention_mask=mask,output_hidden_states=True),dict(
        visual_start=a.shape[1]+2,visual_count=visual.shape[1],sequence_length=inputs.shape[1],
        official_attention_modality_positions=pos.detach().cpu().tolist())


@contextmanager
def mmu_loop_context(runtime,operator,span):
    if operator.condition is None:raise ValueError('Missing current-state MMU evidence condition')
    start,count,length=(span[k] for k in ('visual_start','visual_count','sequence_length'))
    if (any(isinstance(x,bool) or not isinstance(x,int) for x in (start,count,length)) or
            start<1 or count<1 or start+count>length or math.isqrt(count)**2!=count):
        raise ValueError('Invalid explicit MMU visual span')
    previous_positions=operator.positions;previous_enabled=operator.enabled
    try:
        # Internal adapter layout only. Official attention mask is never rewritten.
        operator.positions=torch.tensor([[[start-1,count+1]]],device=runtime.device)
        with install(runtime,operator):
            operator.enabled=True
            yield
    finally:
        operator.positions=previous_positions;operator.enabled=previous_enabled
