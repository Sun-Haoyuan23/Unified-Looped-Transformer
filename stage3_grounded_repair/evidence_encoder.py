"""Target/opposite MMU states, separate margins, and compact evidence encoding."""
import sys
import torch
from torch import nn


class EvidenceEncoder(nn.Module):
    def __init__(self, level=1, semantic_size=1536, width=128):
        super().__init__()
        self.level,self.semantic_size=level,semantic_size
        size={0:2,1:4,2:4+3*semantic_size}[level]
        self.network=nn.Sequential(nn.LayerNorm(size),nn.Linear(size,width),nn.SiLU())
        self.output_size=width

    def forward(self, condition):
        if self.level==0:
            value=condition[:,[2,3]]
        elif self.level==1:
            value=condition[:,:4]
        else:
            value=condition
        return self.network(value.float())


def condition_tensor(evidence, true_t, device):
    scalars=torch.tensor([evidence['target_margin']/8,evidence['opposite_margin']/8,
                          evidence['evidence']/8,true_t],device=device).float()
    target=evidence['target_hidden'].to(device).float()
    opposite=evidence['opposite_hidden'].to(device).float()
    target=torch.nn.functional.normalize(target,dim=-1)
    opposite=torch.nn.functional.normalize(opposite,dim=-1)
    return torch.cat((scalars,target,opposite,target-opposite)).unsqueeze(0)


class EvidenceReader:
    def __init__(self, runtime):
        from mmu_evidence_reader import MMUEvidenceReader
        self.runtime=runtime
        self.base=MMUEvidenceReader(runtime.model,runtime.tokenizer,runtime.ids,
                                    runtime.nmmu,runtime.device)

    def _question(self, visual, question):
        r=self.runtime
        tok=r.tokenizer
        system=tok('system\nYou are a helpful assistant.<|im_end|>',add_special_tokens=False).input_ids
        user=tok('\n<|im_start|>user\n',add_special_tokens=False).input_ids
        assistant=tok('\n<|im_start|>assistant\n',add_special_tokens=False).input_ids
        query=tok(question,add_special_tokens=False).input_ids
        a=torch.tensor([r.ids['bos_id']]+system+user,device=r.device)[None]
        b=torch.tensor([r.ids['boi_id'],r.ids['eoi_id']]+query+assistant,device=r.device)[None]
        ae=r.model.showo.model.embed_tokens(a); be=r.model.showo.model.embed_tokens(b)
        time=r.model.time_embed(torch.tensor([[1.]],device=r.device),ae.dtype)
        if hasattr(r.model,'time_embed_proj'): time=r.model.time_embed_proj(time)
        inputs=torch.cat((ae,be[:,:1],time,visual,be[:,1:]),dim=1)
        pos=torch.tensor([a.shape[1]+2,r.nmmu],device=r.device)[None,None]
        mask=r.mask_fn(B=1,LEN=inputs.shape[1],modalities=pos,device=r.device,
                       inverted=True).to(inputs.dtype)
        output=r.model.showo(inputs_embeds=inputs,attention_mask=mask,
                             output_hidden_states=True)
        logits=output['logits'][0,-1].float()
        margin=torch.logsumexp(logits[self.base.yes_ids],0)-torch.logsumexp(logits[self.base.no_ids],0)
        return float(margin),output['hidden_states'][-1][0,-1].detach().cpu()

    @torch.no_grad()
    def read(self, clean, prompt):
        from mmu_evidence_reader import RELATION, OPPOSITE
        relation=RELATION.match(prompt)
        if relation is None: raise ValueError('First oracle is limited to explicit spatial relations')
        subject,expected,reference=relation.groups(); expected=expected.lower()
        visual=self.base._visual_features(clean)
        questions=[f'In this image, is the {subject} {rel} the {reference}? Answer only Yes or No.'
                   for rel in (expected,OPPOSITE[expected])]
        scores=[self._question(visual,q) for q in questions]
        return {'target_question':questions[0],'opposite_question':questions[1],
            'target_margin':scores[0][0],'opposite_margin':scores[1][0],
            'evidence':scores[0][0]-scores[1][0],
            'target_hidden':scores[0][1],'opposite_hidden':scores[1][1],
            'subject':subject,'reference':reference,'relation':expected}
