"""Measure the supported-operator FLOPs and latency of one MMU check.

Use the same 432px Show-o2 model and a spatial counterfactual query as the
in-trajectory controller. This is a relative profiler proxy, not a full
end-to-end hardware FLOP count.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from mmu_evidence_reader import MMUEvidenceReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gradient-task-id", default="")
    args = parser.parse_args()
    source = args.work / "Show-o" / "show-o2"
    sys.path.insert(0, str(source))
    from models import Showo2Qwen2_5
    from models.misc import get_text_tokenizer
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
    (_ngen, nmmu, _nvideo, _seq, _text, channels, patch,
     latent_w, latent_h, *_rest) = get_hyper_params(config, tokenizer, ids)
    reader = MMUEvidenceReader(model, tokenizer, ids, nmmu, device)
    if args.gradient_task_id:
        model.requires_grad_(False)
        payload = torch.load(
            args.work / "stage2b" / "trajectories" / "val" /
            f"{args.gradient_task_id}.pt", map_location="cpu",
            weights_only=False)
        index = 35
        clean = (payload["zt"][index] +
                 (1 - float(payload["t"][index])) *
                 payload["teacher_velocity"][index]).to(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        result = reader.spatial_margin_gradient(clean, payload["prompt"])
        torch.cuda.synchronize()
        gradient = result.pop("gradient")
        velocity_rms = float(payload["teacher_velocity"][index].float()
                             .square().mean().sqrt())
        gradient_rms = float(gradient.float().square().mean().sqrt())
        responses = []
        for strength in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
            corrected_clean = (clean.float() +
                               (1 - float(payload["t"][index])) * strength *
                               velocity_rms * gradient / max(gradient_rms, 1e-8))
            responses.append({"strength": strength,
                              "margin": reader.read(corrected_clean,
                                                    payload["prompt"])["margin"]})
        report = {"task_id": args.gradient_task_id,
                  "t": float(payload["t"][index]),
                  "runtime_s": time.perf_counter() - started,
                  "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
                  "gradient_rms": gradient_rms,
                  "velocity_rms": velocity_rms,
                  "local_responses": responses,
                  "gradient_max_abs": float(gradient.float().abs().max()),
                  **result}
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)
        return
    clean = torch.randn((1, channels, latent_h * patch, latent_w * patch),
                        device=device, dtype=dtype)
    prompt = "a photo of a zebra below a broccoli"
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        reader.read(clean, prompt)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                with_flops=True) as profile:
            evidence = reader.read(clean, prompt)
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    result = {
        "check": "spatial_relation_vs_opposite_direct_clean_latent",
        "profiled_operator_flops": int(sum(e.flops for e in profile.key_averages()
                                           if e.flops)),
        "profiled_runtime_s": elapsed,
        "evidence": evidence,
        "note": "PyTorch profiler reports supported operators only",
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
