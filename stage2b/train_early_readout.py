"""Train an ELT-style intermediate exit on a fixed two-loop visual model."""

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

from early_readout import EarlyReadout, save_readout
from elt_model import install_svd_loop_head, load_head_checkpoint
from train_elt_onpolicy import FirstLoopCapture, forward_exits, prepare_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpu-physical", type=int, required=True)
    parser.add_argument("--head-checkpoint", required=True)
    parser.add_argument("--head-rank", type=int, default=1024)
    parser.add_argument("--attn-rank", type=int, default=0)
    parser.add_argument("--mlp-rank", type=int, default=0)
    parser.add_argument("--adaln-rank", type=int, default=0)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
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
    install_svd_loop_head(model, rank=args.head_rank, trainable=False,
                          attention_rank=args.attn_rank,
                          mlp_rank=args.mlp_rank,
                          adaln_rank=args.adaln_rank)
    load_head_checkpoint(model, args.work / "stage2b" / args.head_checkpoint)
    model._elt_early_readout = EarlyReadout(
        model.diffusion_head_config.hidden_size, args.width).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = list(model._elt_early_readout.parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    capture = FirstLoopCapture(model, 4)
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
        payload = torch.load(args.work / "stage2b" / "trajectories" / "train" /
                             f"{job['id']}.pt", map_location="cpu", weights_only=False)
        records.append((payload, inputs))
    rng = random.Random(args.seed)
    root = args.work / "stage2b"
    metrics = root / f"train_{args.run_tag}.jsonl"
    print(json.dumps({"train_prompts": len(records),
                      "trainable_parameters": sum(p.numel() for p in trainable),
                      "head_checkpoint": args.head_checkpoint}), flush=True)
    step_times = []
    torch.cuda.reset_peak_memory_stats(device)
    with metrics.open("w", encoding="utf-8") as stream:
        for step in range(args.steps):
            payload, inputs = records[rng.randrange(len(records))]
            index = rng.randrange(49)
            zt = payload["zt"][index].to(device)
            target = payload["teacher_velocity"][index].to(device).float()
            lam = 1.0 - step / max(1, args.steps - 1)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            first, final = forward_exits(
                model, capture, inputs, zt, payload["t"][index],
                float(payload["guidance"]))
            loss_gt = F.mse_loss(first, target)
            loss_ilsd = F.mse_loss(first, final.detach())
            loss = lam * loss_gt + (1 - lam) * loss_ilsd
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step+1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - started)
            if step == 0 or (step + 1) % args.log_every == 0:
                row = {"step": step + 1, "loss": float(loss),
                       "loss_gt": float(loss_gt), "loss_ilsd": float(loss_ilsd),
                       "lambda": lam,
                       "median_recent_s": statistics.median(
                           step_times[-args.log_every:]),
                       "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(json.dumps(row), flush=True)
            if args.save_every and (step + 1) % args.save_every == 0:
                path = root / f"{args.run_tag}_step{step+1}.pt"
                save_readout(model._elt_early_readout, path, step+1,
                             args.width, args.head_checkpoint)
                print(json.dumps({"checkpoint": str(path), "step": step+1}), flush=True)
    final = root / f"{args.run_tag}.pt"
    save_readout(model._elt_early_readout, final, args.steps,
                 args.width, args.head_checkpoint)
    print(json.dumps({"checkpoint": str(final), "steps": args.steps,
                      "median_step_s": statistics.median(step_times)}), flush=True)
    capture.close()


if __name__ == "__main__":
    main()
