"""Evaluate both loop exits on unseen on-policy teacher trajectories."""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.nn import functional as F

from elt_model import (install_elt_head, install_residual_tail_head,
                       install_mlp_shared_tail_head, install_lora_mlp_tail_head,
                       install_svd_loop_head, install_svd_sequence_head,
                       install_svd_middle_head, merge_svd_adapters_for_inference,
                       load_head_checkpoint)
from train_elt_onpolicy import FirstLoopCapture, forward_exits, prepare_inputs
from early_readout import EarlyReadout, load_readout
from final_head_tuning import load_final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpu-physical", type=int, required=True)
    parser.add_argument("--indices", default="0,8,16,24,32,40,48")
    parser.add_argument("--checkpoint", default="elt_onpolicy_5x2_bf16.pt")
    parser.add_argument("--tag", default="elt_onpolicy_5x2")
    parser.add_argument("--trajectory-dir", default="val")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--prefix-layers", type=int, default=0)
    parser.add_argument("--unique-layers", type=int, default=5)
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--architecture", choices=("base", "shared", "residual_tail",
                                                   "mlp_shared", "lora_mlp", "svd_loop", "svd_sequence", "svd_middle"),
                        default="shared")
    parser.add_argument("--adapter-width", type=int, default=1536)
    parser.add_argument("--lora-rank", type=int, default=128)
    parser.add_argument("--attn-rank", type=int, default=0)
    parser.add_argument("--mlp-rank", type=int, default=0)
    parser.add_argument("--adaln-rank", type=int, default=0)
    parser.add_argument("--early-checkpoint", default="")
    parser.add_argument("--early-width", type=int, default=4096)
    parser.add_argument("--final-checkpoint", default="")
    parser.add_argument("--merged-adapters", action="store_true")
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose one healthy GPU by UUID")
    indices = [int(x) for x in args.indices.split(",")]
    if any(i < 0 or i >= 49 for i in indices):
        raise ValueError(indices)
    work = args.work
    source = work / "Show-o" / "show-o2"
    sys.path.insert(0, str(source))
    from models import Showo2Qwen2_5, omni_attn_mask_naive
    from models.misc import get_text_tokenizer
    from utils import get_hyper_params

    device, dtype = torch.device("cuda:0"), torch.bfloat16
    config = OmegaConf.load(source / "configs" / "showo2_1.5b_demo_432x432.yaml")
    tokenizer, token_ids = get_text_tokenizer(
        str(work / "models" / "qwen2.5-1.5b-tokenizer"),
        add_showo_tokens=True, return_showo_token_ids=True, llm_name="qwen2_5")
    model = Showo2Qwen2_5.from_pretrained(
        str(work / "models" / "show-o2-1.5B-local"), use_safetensors=False
    ).to(device=device, dtype=dtype).eval()
    if args.architecture == "base":
        pass
    elif args.architecture == "residual_tail":
        install_residual_tail_head(model, prefix_layers=args.prefix_layers,
                                   unique_layers=args.unique_layers,
                                   adapter_width=args.adapter_width,
                                   trainable=False)
    elif args.architecture == "mlp_shared":
        install_mlp_shared_tail_head(model, prefix_layers=args.prefix_layers,
                                     unique_layers=args.unique_layers,
                                     trainable=False)
    elif args.architecture == "lora_mlp":
        install_lora_mlp_tail_head(model, prefix_layers=args.prefix_layers,
                                   unique_layers=args.unique_layers,
                                   rank=args.lora_rank, trainable=False)
    elif args.architecture in ("svd_loop", "svd_sequence", "svd_middle"):
        installer = {"svd_loop": install_svd_loop_head,
                     "svd_sequence": install_svd_sequence_head,
                     "svd_middle": install_svd_middle_head}[args.architecture]
        installer(model, rank=args.lora_rank, trainable=False,
                              attention_rank=args.attn_rank, mlp_rank=args.mlp_rank,
                              adaln_rank=args.adaln_rank)
    else:
        install_elt_head(model, unique_layers=args.unique_layers, loops=args.loops,
                         prefix_layers=args.prefix_layers, precision=torch.float32,
                         trainable=False)
    checkpoint = ({"steps": 0} if args.architecture == "base" else
                  load_head_checkpoint(model, work / "stage2b" / args.checkpoint))
    if args.merged_adapters:
        merge_svd_adapters_for_inference(model)
    if args.final_checkpoint:
        load_final(model, work / "stage2b" / args.final_checkpoint)
    if args.early_checkpoint:
        model._elt_early_readout = EarlyReadout(
            model.diffusion_head_config.hidden_size, args.early_width).to(device)
        load_readout(model._elt_early_readout,
                     work / "stage2b" / args.early_checkpoint)
        model._elt_early_readout.eval()
    capture = FirstLoopCapture(model, args.prefix_layers + args.unique_layers - 1)
    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1
    (nvisual, _nmmu, _nvideo, max_seq_len, max_text_len, _channels, _patch,
     _w, _h, pad_id, bos_id, eos_id, boi_id, eoi_id,
     _bov, _eov, img_pad_id, _vid, _guide) = get_hyper_params(config, tokenizer, token_ids)
    hyper = (nvisual, max_seq_len, max_text_len, pad_id, bos_id, eos_id,
             boi_id, eoi_id, img_pad_id)
    jobs = json.loads(args.manifest.read_text(encoding="utf-8"))[args.split]
    if args.max_prompts:
        jobs = jobs[:args.max_prompts]
    rows = []
    with torch.inference_mode():
        for job in jobs:
            payload = torch.load(work / "stage2b" / "trajectories" / args.trajectory_dir /
                                 f"{job['id']}.pt", map_location="cpu", weights_only=False)
            inputs = prepare_inputs(job, tokenizer, hyper, omni_attn_mask_naive,
                                    device, dtype)
            for index in indices:
                zt = payload["zt"][index].to(device)
                target = payload["teacher_velocity"][index].to(device).float()
                first, final = forward_exits(
                    model, capture, inputs, zt, payload["t"][index],
                    float(payload["guidance"]))
                rows.append({"id": job["id"], "category": job["category"],
                             "time_index": index, "t": float(payload["t"][index]),
                             "mse_L1": float(F.mse_loss(first, target)),
                             "mse_L2": float(F.mse_loss(final, target)),
                             "mse_L1_L2": float(F.mse_loss(first, final))})
            print(json.dumps({"id": job["id"], "states_done": len(rows)}), flush=True)
    root = work / "stage2b"
    with (root / f"val_trajectory_{args.tag}.csv").open(
            "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"prompts": len(jobs), "split": args.split, "states": len(rows),
               "trajectory_dir": args.trajectory_dir,
               "indices": indices, "checkpoint_steps": checkpoint["steps"],
               "mse_L1_mean": statistics.mean(row["mse_L1"] for row in rows),
               "mse_L2_mean": statistics.mean(row["mse_L2"] for row in rows),
               "mse_L1_L2_mean": statistics.mean(row["mse_L1_L2"] for row in rows)}
    (root / f"val_trajectory_summary_{args.tag}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    capture.close()


if __name__ == "__main__":
    main()
