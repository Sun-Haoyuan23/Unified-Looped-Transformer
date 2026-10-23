"""Measure which pretrained head components tolerate loop-wise weight sharing."""

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.nn import functional as F

from train_elt_onpolicy import FirstLoopCapture, forward_exits, prepare_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--gpu-physical", type=int, required=True)
    parser.add_argument("--indices", default="0,24,48")
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one healthy GPU by UUID")
    work = args.work
    source = work / "Show-o" / "show-o2"
    sys.path.insert(0, str(source))
    from models import Showo2Qwen2_5, omni_attn_mask_naive
    from models.misc import get_text_tokenizer
    from utils import get_hyper_params

    device, dtype = torch.device("cuda:0"), torch.bfloat16
    config = OmegaConf.load(source / "configs" / "showo2_1.5b_demo_432x432.yaml")
    tokenizer, ids = get_text_tokenizer(
        str(work / "models" / "qwen2.5-1.5b-tokenizer"),
        add_showo_tokens=True, return_showo_token_ids=True, llm_name="qwen2_5")
    model = Showo2Qwen2_5.from_pretrained(
        str(work / "models" / "show-o2-1.5B-local"), use_safetensors=False
    ).to(device=device, dtype=dtype).eval()
    original_tail = [copy.deepcopy(model.diffusion_head_a[i]) for i in (8, 9)]
    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1
    (nvisual, _nmmu, _nvideo, max_seq_len, max_text_len, _channels, _patch,
     _w, _h, pad_id, bos_id, eos_id, boi_id, eoi_id,
     _bov, _eov, img_pad_id, _vid, _guide) = get_hyper_params(config, tokenizer, ids)
    hyper = (nvisual, max_seq_len, max_text_len, pad_id, bos_id, eos_id,
             boi_id, eoi_id, img_pad_id)
    jobs = json.loads((work / "stage2b" / "manifest.json").read_text())["val"]
    records = []
    for job in jobs:
        inputs = prepare_inputs(job, tokenizer, hyper, omni_attn_mask_naive,
                                device, dtype)
        payload = torch.load(work / "stage2b" / "trajectories" / "val" /
                             f"{job['id']}.pt", map_location="cpu", weights_only=False)
        records.append((job, inputs, payload))
    components = {
        "baseline": (),
        "mlp_only": ("mlp",),
        "attention_only": ("self_attn",),
        "adaln_only": ("adaLN_modulation",),
        "norms_only": ("input_layernorm", "post_attention_layernorm"),
        "mlp_and_adaln": ("mlp", "adaLN_modulation"),
        "attention_and_adaln": ("self_attn", "adaLN_modulation"),
    }
    result = {}
    indices = [int(x) for x in args.indices.split(",")]
    with torch.inference_mode():
        for variant, names in components.items():
            for i in range(2):
                model.diffusion_head_a[8 + i] = copy.deepcopy(original_tail[i])
                for name in names:
                    setattr(model.diffusion_head_a[8 + i], name,
                            getattr(model.diffusion_head_a[6 + i], name))
            capture = FirstLoopCapture(model, 7)
            errors = []
            for job, inputs, payload in records:
                for index in indices:
                    zt = payload["zt"][index].to(device)
                    _, final = forward_exits(model, capture, inputs, zt,
                                             payload["t"][index],
                                             float(payload["guidance"]))
                    target = payload["teacher_velocity"][index].to(device).float()
                    errors.append(float(F.mse_loss(final, target)))
            capture.close()
            result[variant] = {"mse_mean": statistics.mean(errors),
                               "mse_median": statistics.median(errors),
                               "states": len(errors), "shared_components": names}
            print(json.dumps({"variant": variant, **result[variant]}), flush=True)
    output = work / "stage2b" / "component_tie_screen.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
