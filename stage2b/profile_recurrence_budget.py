"""Measure unique parameters and operator FLOPs for Show-o2 loop candidates.

Profiler FLOPs cover supported PyTorch operators only; report them as a
reproducible relative proxy, never as complete end-to-end generation FLOPs.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from elt_model import _delta_loop_block


def count(module):
    return sum(p.numel() for p in module.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.work / "Show-o" / "show-o2"
    sys.path.insert(0, str(source))
    from models import Showo2Qwen2_5, omni_attn_mask_naive
    from models.misc import get_text_tokenizer, prepare_gen_input
    from utils import get_hyper_params

    device, dtype = torch.device("cuda:0"), torch.bfloat16
    config = OmegaConf.load(source / "configs" / "showo2_1.5b_demo_432x432.yaml")
    tokenizer, ids = get_text_tokenizer(
        str(args.work / "models" / "qwen2.5-1.5b-tokenizer"),
        add_showo_tokens=True, return_showo_token_ids=True, llm_name="qwen2_5")
    model = Showo2Qwen2_5.from_pretrained(
        str(args.work / "models" / "show-o2-1.5B-local"),
        use_safetensors=False).to(device=device, dtype=dtype).eval()
    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1
    (nvisual, _nmmu, _nvideo, max_seq_len, max_text_len, channels, patch,
     latent_w, latent_h, pad_id, bos_id, eos_id, boi_id, eoi_id,
     _bov, _eov, img_pad_id, _vid, _guidance) = get_hyper_params(config, tokenizer, ids)
    cond, uncond, cpos, upos = prepare_gen_input(
        ["a photo of a zebra below a broccoli"], tokenizer, nvisual, bos_id,
        eos_id, boi_id, eoi_id, pad_id, img_pad_id, max_text_len, device)
    tokens = torch.cat((cond, uncond))
    positions = torch.cat((cpos, upos))
    mask = omni_attn_mask_naive(2, max_seq_len, positions, device).to(dtype)
    generator = torch.Generator(device=device).manual_seed(101)
    noise = torch.randn((1, channels, latent_h * patch, latent_w * patch),
                        device=device, dtype=dtype, generator=generator)
    z = torch.cat((noise, noise))
    t = torch.full((2,), 0.65, device=device)
    kwargs = dict(text_tokens=tokens, attention_mask=mask,
                  modality_positions=positions, output_hidden_states=True,
                  max_seq_len=max_seq_len, guidance_scale=5.0)
    original = list(model.diffusion_head_a)
    single_delta_256 = _delta_loop_block(original[1], original[2],
                                         256, 256, 256, False)
    structures = {
        "released_10_unique_10_calls": original,
        "shared_middle_7_unique_10_calls": original[:5] + original[2:5] + original[8:],
        "shared_middle_7_unique_7_calls": original[:5] + original[8:],
        "shared_middle_6_unique_8_calls": original[:4] + original[2:4] + original[8:],
        "shared_single_9_unique_9_calls": original[:2] + original[3:],
        "shared_single_9_unique_10_calls": original[:2] + [original[1]] + original[3:],
        "svd_single_r256_9_exit": original[:2] + original[3:],
        "svd_single_r256_10_exit": original[:2] + [single_delta_256] + original[3:],
    }
    rows = []
    svd_full_head = nn.ModuleList(original[:2] + [single_delta_256] + original[3:])
    svd_unique_head_parameters = count(svd_full_head)
    released_head_parameters = count(nn.ModuleList(original))
    released_model_parameters = count(model)
    for name, blocks in structures.items():
        model.diffusion_head_a = nn.ModuleList(blocks)
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            model.t2i_generate(z, t, **kwargs)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                     torch.profiler.ProfilerActivity.CUDA],
                                        with_flops=True) as profile:
                model.t2i_generate(z, t, **kwargs)
                torch.cuda.synchronize()
        op_flops = sum(event.flops for event in profile.key_averages() if event.flops)
        actual_head_parameters = (svd_unique_head_parameters
                                  if name.startswith("svd_single_r256") else
                                  count(model.diffusion_head_a))
        actual_total_parameters = (released_model_parameters -
                                   released_head_parameters +
                                   actual_head_parameters)
        row = {"name": name, "total_unique_parameters": actual_total_parameters,
               "visual_head_unique_parameters": actual_head_parameters,
               "visual_block_calls_per_velocity": len(blocks),
               "profiled_operator_flops_per_velocity": int(op_flops),
               "note": "PyTorch profiler reports supported operators only"}
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
