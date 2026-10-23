"""Distill two teacher Euler evaluations into one two-loop visual prediction."""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.nn import functional as F

from loop_solver_model import install_loop_solver, save_loop_adapter
from train_elt_onpolicy import guide, prepare_inputs


def paired_velocity(payload, index):
    times = payload["t"]
    first = payload["teacher_velocity"][index].float()
    second = payload["teacher_velocity"][index + 1].float()
    dt_first = abs(float(times[index + 1]) - float(times[index]))
    dt_second = abs(float(times[index + 2]) - float(times[index + 1]))
    return ((dt_first * first + dt_second * second) /
            (dt_first + dt_second))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpu-physical", type=int, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--anchor-weight", type=float, default=0.1)
    parser.add_argument("--run-tag", default="loop_solver_w128")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose one healthy GPU by UUID")
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
    refine = install_loop_solver(model, width=args.width, trainable=True)
    trainable = list(refine.adapter.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                  betas=(0.9, 0.99), weight_decay=0.01)
    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1
    (nvisual, _nmmu, _nvideo, max_seq_len, max_text_len, _channels, _patch,
     _w, _h, pad_id, bos_id, eos_id, boi_id, eoi_id,
     _bov, _eov, img_pad_id, _vid, _guide) = get_hyper_params(config, tokenizer, ids)
    hyper = (nvisual, max_seq_len, max_text_len, pad_id, bos_id, eos_id,
             boi_id, eoi_id, img_pad_id)
    jobs = json.loads(args.manifest.read_text(encoding="utf-8"))["train"]
    records = []
    for job in jobs:
        inputs = prepare_inputs(job, tokenizer, hyper, omni_attn_mask_naive,
                                device, dtype)
        payload = torch.load(work / "stage2b" / "trajectories" / "train" /
                             f"{job['id']}.pt", map_location="cpu", weights_only=False)
        if payload["zt"].shape[0] != 49:
            raise RuntimeError(f"Invalid trajectory {job['id']}")
        records.append((payload, inputs))
    rng = random.Random(20261003)
    output = work / "stage2b"
    metrics = output / f"train_{args.run_tag}.jsonl"
    print(json.dumps({"train_prompts": len(records),
                      "paired_teacher_intervals": len(records) * 24,
                      "trainable_parameters": sum(p.numel() for p in trainable),
                      "architecture": "original ten visual blocks repeated twice",
                      "flow_calls_at_inference": 24,
                      "head_block_calls_at_inference": 480,
                      "baseline_head_block_calls": 490}), flush=True)
    times = []
    torch.cuda.reset_peak_memory_stats(device)
    with metrics.open("w", encoding="utf-8") as stream:
        for step in range(args.steps):
            payload, inputs = records[rng.randrange(len(records))]
            index = 2 * rng.randrange(24)
            zt = payload["zt"][index].to(device)
            t_value = float(payload["t"][index])
            target = paired_velocity(payload, index).to(device)
            instant = payload["teacher_velocity"][index].float().to(device)
            tokens, positions, mask = inputs
            zz = torch.cat((zt, zt), dim=0)
            t = torch.full((2,), t_value, device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=dtype):
                _logits, raw = model(
                    text_tokens=tokens, image_latents=zz, t=t,
                    attention_mask=mask, modality_positions=positions,
                    output_hidden_states=True, max_seq_len=mask.shape[-1],
                    device=device)
            guided = guide(raw.float(), float(payload["guidance"]))
            loss_pair = F.mse_loss(guided, target)
            loss_anchor = F.mse_loss(guided, instant)
            loss = loss_pair + args.anchor_weight * loss_anchor
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step+1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
            if step == 0 or (step + 1) % args.log_every == 0:
                row = {"step": step + 1, "loss": round(float(loss), 6),
                       "paired_mse": round(float(loss_pair), 6),
                       "instant_mse": round(float(loss_anchor), 6),
                       "median_recent_s": round(statistics.median(times[-args.log_every:]), 4),
                       "peak_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 3)}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(json.dumps(row), flush=True)
            if args.save_every and (step + 1) % args.save_every == 0:
                save_loop_adapter(refine, output / f"{args.run_tag}_step{step+1}_bf16.pt",
                                  step + 1, args.width)
    save_loop_adapter(refine, output / f"{args.run_tag}_bf16.pt",
                      args.steps, args.width)


if __name__ == "__main__":
    main()
