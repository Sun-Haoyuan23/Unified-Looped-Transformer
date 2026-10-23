"""Fit the compact loop's 8.5M prediction head to teacher on-policy velocities."""

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

from elt_model import install_svd_loop_head, load_head_checkpoint
from final_head_tuning import install_tunable_final, save_final
from train_elt_onpolicy import FirstLoopCapture, forward_exits, prepare_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpu-physical", type=int, required=True)
    parser.add_argument("--loop-checkpoint", required=True)
    parser.add_argument("--rank", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose one healthy GPU by UUID")
    torch.set_float32_matmul_precision("high")
    source = args.work / "Show-o" / "show-o2"
    sys.path.insert(0, str(source))
    from models import Showo2Qwen2_5, omni_attn_mask_naive
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
    install_svd_loop_head(model, rank=args.rank, trainable=False)
    load_head_checkpoint(model, args.work / "stage2b" / args.loop_checkpoint)
    trainable = install_tunable_final(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in trainable:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                  betas=(0.9, 0.99), weight_decay=0.01)
    capture = FirstLoopCapture(model, 4)
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
        payload = torch.load(args.work / "stage2b" / "trajectories" / "train" /
                             f"{job['id']}.pt", map_location="cpu", weights_only=False)
        records.append((payload, inputs))
    rng = random.Random(args.seed)
    root = args.work / "stage2b"
    metrics = root / f"train_{args.run_tag}.jsonl"
    print(json.dumps({"train_prompts": len(records),
                      "trainable_parameters": sum(p.numel() for p in trainable),
                      "loop_checkpoint": args.loop_checkpoint}), flush=True)
    step_times = []
    torch.cuda.reset_peak_memory_stats(device)
    with metrics.open("w", encoding="utf-8") as stream:
        for step in range(args.steps):
            payload, inputs = records[rng.randrange(len(records))]
            index = rng.randrange(49)
            zt = payload["zt"][index].to(device)
            target = payload["teacher_velocity"][index].to(device).float()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            _first, final = forward_exits(
                model, capture, inputs, zt, payload["t"][index],
                float(payload["guidance"]))
            loss = F.mse_loss(final, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step+1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - started)
            if step == 0 or (step + 1) % args.log_every == 0:
                row = {"step": step+1, "loss": float(loss),
                       "median_recent_s": statistics.median(step_times[-args.log_every:]),
                       "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(json.dumps(row), flush=True)
            if args.save_every and (step+1) % args.save_every == 0:
                path = root / f"{args.run_tag}_step{step+1}.pt"
                save_final(model, path, step+1, args.loop_checkpoint)
                print(json.dumps({"checkpoint": str(path), "step": step+1}), flush=True)
    final = root / f"{args.run_tag}.pt"
    save_final(model, final, args.steps, args.loop_checkpoint)
    print(json.dumps({"checkpoint": str(final), "steps": args.steps,
                      "median_step_s": statistics.median(step_times)}), flush=True)
    capture.close()


if __name__ == "__main__":
    main()
