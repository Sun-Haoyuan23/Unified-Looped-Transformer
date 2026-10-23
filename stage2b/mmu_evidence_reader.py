"""Read Show-o2 understanding evidence directly from a predicted clean latent.

This avoids VAE decoding/re-encoding in the generation loop. The spatial
counterfactual subtracts an opposite-relation score to cancel yes bias.
"""

import re

import torch


RELATION = re.compile(
    r"^a photo of (?:a|an) (.+?) (left of|right of|above|below) (?:a|an) (.+)$",
    re.IGNORECASE,
)
OPPOSITE = {"above": "below", "below": "above",
            "left of": "right of", "right of": "left of"}


class MMUEvidenceReader:
    def __init__(self, model, tokenizer, token_ids, nvisual, device):
        self.model = model
        self.tokenizer = tokenizer
        self.token_ids = token_ids
        self.nvisual = nvisual
        self.device = device
        self.yes_ids = sorted({tokenizer(s, add_special_tokens=False).input_ids[0]
                               for s in ("Yes", " yes", "YES", "yes")})
        self.no_ids = sorted({tokenizer(s, add_special_tokens=False).input_ids[0]
                              for s in ("No", " no", "NO", "no")})
        self.yes_ids = sorted(set(self.yes_ids) - set(self.no_ids))

    def _question_margin(self, visual, question, return_tensor=False):
        from models import omni_attn_mask_naive
        tok = self.tokenizer
        system = tok("system\nYou are a helpful assistant.<|im_end|>",
                     add_special_tokens=False).input_ids
        user = tok("\n<|im_start|>user\n", add_special_tokens=False).input_ids
        assistant = tok("\n<|im_start|>assistant\n", add_special_tokens=False).input_ids
        query = tok(question, add_special_tokens=False).input_ids
        a = torch.tensor([self.token_ids["bos_id"]] + system + user,
                         device=self.device)[None]
        b = torch.tensor([self.token_ids["boi_id"], self.token_ids["eoi_id"]]
                         + query + assistant, device=self.device)[None]
        ae = self.model.showo.model.embed_tokens(a)
        be = self.model.showo.model.embed_tokens(b)
        time = self.model.time_embed(torch.tensor([[1.0]], device=self.device), ae.dtype)
        if hasattr(self.model, "time_embed_proj"):
            time = self.model.time_embed_proj(time)
        inputs = torch.cat((ae, be[:, :1], time, visual, be[:, 1:]), dim=1)
        positions = torch.tensor([a.shape[1] + 2, self.nvisual],
                                 device=self.device)[None, None]
        mask = omni_attn_mask_naive(B=1, LEN=inputs.shape[1], modalities=positions,
                                    device=self.device, inverted=True).to(inputs.dtype)
        logits = self.model.showo(inputs_embeds=inputs, attention_mask=mask)["logits"][0, -1].float()
        margin = torch.logsumexp(logits[self.yes_ids], dim=0) - torch.logsumexp(logits[self.no_ids], dim=0)
        return margin if return_tensor else float(margin)

    def _visual_features(self, clean_latent):
        latent = clean_latent.to(dtype=next(self.model.parameters()).dtype)
        und = self.model.image_embedder_und(latent)
        gen = self.model.image_embedder_gen(latent)
        und = und + self.model.position_embedding(self.model.image_position_ids)
        und = self.model.und_trans(und)["last_hidden_state"]
        return self.model.fusion_proj(torch.cat((und, gen), dim=-1))

    @torch.inference_mode()
    def read(self, clean_latent, prompt):
        visual = self._visual_features(clean_latent)
        relation = RELATION.match(prompt)
        if relation:
            subject, expected, reference = relation.groups()
            expected = expected.lower()
            question = (f"In this image, is the {subject} {expected} the {reference}? "
                        "Answer only Yes or No.")
            opposite = (f"In this image, is the {subject} {OPPOSITE[expected]} the {reference}? "
                        "Answer only Yes or No.")
            preferred = self._question_margin(visual, question)
            counterfactual = self._question_margin(visual, opposite)
            return {"mode": "relation_counterfactual", "margin": preferred - counterfactual,
                    "preferred_margin": preferred, "opposite_margin": counterfactual}
        question = ("Does the image accurately depict this description: "
                    f"{prompt} Answer only Yes or No.")
        return {"mode": "prompt", "margin": self._question_margin(visual, question)}

    def spatial_margin_gradient(self, clean_latent, prompt):
        """Offline oracle: gradient of the target-vs-opposite MMU evidence.

        Separate forwards cap activation memory. This is deliberately excluded
        from the deployed controller, which only evaluates evidence.
        """
        relation = RELATION.match(prompt)
        if relation is None:
            raise ValueError("Spatial gradient requires a parsed relation")
        subject, expected, reference = relation.groups()
        expected = expected.lower()
        scores, gradients = [], []
        for candidate in (expected, OPPOSITE[expected]):
            latent = clean_latent.detach().float().requires_grad_(True)
            question = (f"In this image, is the {subject} {candidate} the {reference}? "
                        "Answer only Yes or No.")
            with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                margin = self._question_margin(
                    self._visual_features(latent), question,
                    return_tensor=True)
                gradient, = torch.autograd.grad(margin, latent)
            scores.append(float(margin.detach()))
            gradients.append(gradient.detach())
        return {"margin": scores[0] - scores[1],
                "preferred_margin": scores[0],
                "opposite_margin": scores[1],
                "gradient": gradients[0] - gradients[1]}
