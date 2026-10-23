"""Verify the generation-head loop leaves Show-o2's understanding logits exact."""

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from elt_model import install_elt_head, load_head_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--checkpoint", default="elt_onpolicy_5x2_bf16.pt")
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose one healthy GPU by UUID")
    work = args.work
    source = work / "Show-o" / "show-o2"
    sys.path.insert(0, str(source))
    from models import Showo2Qwen2_5, omni_attn_mask_naive
    from models.misc import get_text_tokenizer, prepare_gen_input
    from utils import get_hyper_params

    device, dtype = torch.device("cuda:0"), torch.bfloat16
    config = OmegaConf.load(source / "configs" / "showo2_1.5b_demo_432x432.yaml")
    tokenizer, token_ids = get_text_tokenizer(
        str(work / "models" / "qwen2.5-1.5b-tokenizer"),
        add_showo_tokens=True, return_showo_token_ids=True, llm_name="qwen2_5")
    model = Showo2Qwen2_5.from_pretrained(
        str(work / "models" / "show-o2-1.5B-local"), use_safetensors=False
    ).to(device=device, dtype=dtype).eval()
    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1
    (nvisual, _nmmu, _nvideo, max_seq_len, max_text_len, _channels, _patch,
     _w, _h, pad_id, bos_id, eos_id, boi_id, eoi_id,
     _bov, _eov, img_pad_id, _vid, _guide) = get_hyper_params(
        config, tokenizer, token_ids)
    payload = torch.load(work / "stage2b" / "trajectories" / "train" /
                         "p0000.pt", map_location="cpu", weights_only=False)
    cond, _uncond, cpos, _upos = prepare_gen_input(
        ["Describe this image."], tokenizer, nvisual, bos_id, eos_id,
        boi_id, eoi_id, pad_id, img_pad_id, max_text_len, device)
    mask = omni_attn_mask_naive(1, max_seq_len, cpos, device).to(dtype)
    kwargs = dict(text_tokens=cond,
                  image_latents=payload["final_latent"].to(device),
                  t=torch.zeros(1, device=device, dtype=torch.float32),
                  attention_mask=mask, modality_positions=cpos,
                  output_hidden_states=True, max_seq_len=max_seq_len,
                  device=device)
    with torch.inference_mode():
        original = model.forward_und_only(**kwargs)
    install_elt_head(model, unique_layers=5, loops=2,
                     precision=torch.float32, trainable=False)
    load_head_checkpoint(model, work / "stage2b" / args.checkpoint)
    with torch.inference_mode():
        modified = model.forward_und_only(**kwargs)
    maximum = float((original.float() - modified.float()).abs().max())
    result = {"checkpoint": args.checkpoint, "max_logit_abs_difference": maximum,
              "exact_match": bool(torch.equal(original, modified)),
              "shape": list(original.shape),
              "scope": "one deterministic image/question input; verifies pathway isolation"}
    (work / "stage2b" / "understanding_path_check.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
