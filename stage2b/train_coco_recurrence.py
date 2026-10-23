"""True image-latent flow training for Show-o2's shared visual core.

The pilot trains recurrent second-visit deltas on disjoint COCO-2017 train
images, with a frozen released-model velocity anchor and a supervised
first-visit exit. Unlike trajectory imitation, its main target is a real image.
"""

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.nn import functional as F

from elt_model import (install_svd_middle_head, install_svd_single_head,
                       load_head_checkpoint)
from train_elt_onpolicy import (FirstLoopCapture, prepare_inputs,
                                 teacher_head_outputs, to_velocity)


def load_record(row, root, device):
    payload = torch.load(root / "latents" / f"{row['id']:012d}.pt",
                         map_location="cpu", weights_only=False)
    if payload["id"] != row["id"] or payload["prompt"] != row["prompt"]:
        raise RuntimeError("Latent cache and manifest disagree")
    return payload["z"].to(device)


def forward_pair(model, capture, teacher_layers, teacher_final, inputs,
                 zt, time_value, early_suffix_start=8):
    tokens, positions, mask = inputs
    capture.states.clear()
    capture.initial_state = None
    t = torch.full((2,), float(time_value), device=zt.device,
                   dtype=torch.float32)
    paired = torch.cat((zt, zt), dim=0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _logits, raw_final = model(
            text_tokens=tokens, image_latents=paired, t=t,
            attention_mask=mask, modality_positions=positions,
            output_hidden_states=True, max_seq_len=mask.shape[-1],
            device=zt.device)
        if len(capture.states) != 2:
            raise RuntimeError(f"Expected two loop exits, got {len(capture.states)}")
        time_embeds = model.time_embed(t, torch.bfloat16)
        position_ids = torch.arange(capture.states[0].shape[1],
                                    device=zt.device).unsqueeze(0)
        hidden = capture.states[0]
        # The shared post-loop blocks make this a genuine intermediate exit.
        # They are run only during training, like Looped-DiT deep supervision.
        for layer in model.diffusion_head_a[early_suffix_start:]:
            hidden = layer(hidden_states=hidden, adaln_input=time_embeds,
                           attention_mask=mask, position_ids=position_ids,
                           modality_positions=positions)[0]
        channels, height, width = zt.shape[1:]
        raw_early = to_velocity(model, hidden, time_embeds, positions,
                                channels, height, width)
    teacher_raw, _ = teacher_head_outputs(
        model, capture, teacher_layers, teacher_final, inputs,
        zt, float(time_value))
    return raw_final.float(), raw_early.float(), teacher_raw.float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--gpu-physical", type=int, required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--architecture", choices=("svd_middle", "svd_single"),
                        default="svd_middle")
    parser.add_argument("--shared-source-index", type=int, default=1)
    parser.add_argument("--shared-target-index", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=256)
    parser.add_argument("--train-scope", choices=("all", "revisit_only"),
                        default="all")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--rank", type=int, default=1280)
    parser.add_argument("--attn-rank", type=int, default=1280)
    parser.add_argument("--mlp-rank", type=int, default=2048)
    parser.add_argument("--adaln-rank", type=int, default=2048)
    parser.add_argument("--early-weight", type=float, default=0.1)
    parser.add_argument("--anchor-weight", type=float, default=0.2)
    parser.add_argument("--uncond-weight", type=float, default=0.1)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=300)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-checkpoints", nargs="*", default=[])
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose one healthy GPU by UUID")
    torch.set_float32_matmul_precision("high")
    source = args.work / "Show-o" / "show-o2"
    sys.path.insert(0, str(source))
    from models import Showo2Qwen2_5, omni_attn_mask_naive
    from models.misc import get_text_tokenizer
    from transport import create_transport
    from utils import get_hyper_params

    device = torch.device("cuda:0")
    config = OmegaConf.load(source / "configs" / "showo2_1.5b_demo_432x432.yaml")
    tokenizer, ids = get_text_tokenizer(
        str(args.work / "models" / "qwen2.5-1.5b-tokenizer"),
        add_showo_tokens=True, return_showo_token_ids=True, llm_name="qwen2_5")
    model = Showo2Qwen2_5.from_pretrained(
        str(args.work / "models" / "show-o2-1.5B-local"),
        use_safetensors=False).to(device=device, dtype=torch.bfloat16).eval()
    import copy
    teacher_layers = copy.deepcopy(model.diffusion_head_a).eval()
    teacher_final = copy.deepcopy(model.diffusion_head_b).eval()
    teacher_layers.requires_grad_(False)
    teacher_final.requires_grad_(False)
    if args.architecture == "svd_single":
        trainable = install_svd_single_head(
            model, args.shared_source_index, args.shared_target_index,
            rank=args.lora_rank, svd_init=True, trainable=True)
        first_exit_index = args.shared_source_index
        early_suffix_start = args.shared_target_index + 1
    else:
        trainable = install_svd_middle_head(
            model, rank=args.rank, trainable=True,
            attention_rank=args.attn_rank, mlp_rank=args.mlp_rank,
            adaln_rank=args.adaln_rank)
        first_exit_index = 4
        early_suffix_start = 8
    checkpoint = load_head_checkpoint(
        model, args.work / "stage2b" / args.init_checkpoint)
    if args.train_scope == "revisit_only":
        if args.architecture != "svd_single":
            raise ValueError("Revisit-only training requires svd_single")
        for parameter in model.diffusion_head_a[args.shared_source_index].parameters():
            parameter.requires_grad_(False)
        trainable = [parameter for parameter in model.parameters()
                     if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable revisit delta parameters")
    capture = FirstLoopCapture(model, first_exit_index=first_exit_index)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                  betas=(0.9, 0.99), weight_decay=0)

    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1
    (nvisual, _nmmu, _nvideo, max_seq_len, max_text_len, _channels, _patch,
     _w, _h, pad_id, bos_id, eos_id, boi_id, eoi_id,
     _bov, _eov, img_pad_id, _vid, _guide) = get_hyper_params(
        config, tokenizer, ids)
    hyper = (nvisual, max_seq_len, max_text_len, pad_id, bos_id, eos_id,
             boi_id, eoi_id, img_pad_id)
    transport = create_transport(
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight,
        train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps,
        snr_type=config.transport.snr_type,
        do_shift=config.transport.do_shift, seq_len=nvisual)
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    split = "val" if args.eval_only else "train"
    records = [row for row in manifest["records"] if row["split"] == split]
    if args.max_records:
        records = records[:args.max_records]
    missing = [row["id"] for row in records
               if not (args.data / "latents" / f"{row['id']:012d}.pt").exists()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} cached {split} latents")
    rng = random.Random(20261003)
    output = args.work / "stage2b"
    metrics_path = output / f"train_{args.run_tag}.jsonl"
    if args.eval_only:
        candidates = [args.init_checkpoint] + args.eval_checkpoints
        results = []
        for filename in candidates:
            load_head_checkpoint(model, output / filename)
            per_image = []
            with torch.no_grad():
                for row in records:
                    z = load_record(row, args.data, device)
                    torch.manual_seed(20261003 + row["id"])
                    t, noise, clean = transport.sample(z)
                    t, zt, target = transport.path_sampler.plan(t, noise, clean)
                    inputs = prepare_inputs(row, tokenizer, hyper,
                                            omni_attn_mask_naive,
                                            device, torch.bfloat16)
                    final, early, teacher = forward_pair(
                        model, capture, teacher_layers, teacher_final,
                        inputs, zt, float(t[0]), early_suffix_start)
                    target = target.float()
                    per_image.append({
                        "id": row["id"], "t": float(t[0]),
                        "final": float(F.mse_loss(final[:1], target)),
                        "early": float(F.mse_loss(early[:1], target)),
                        "teacher": float(F.mse_loss(teacher[:1], target)),
                        "velocity_gap": float(F.mse_loss(final[:1], teacher[:1]))})
            result = {"checkpoint": filename, "n": len(per_image),
                      "means": {key: statistics.mean(x[key] for x in per_image)
                                for key in ("final", "early", "teacher", "velocity_gap")},
                      "fraction_better_than_teacher": sum(
                          x["final"] < x["teacher"] for x in per_image) / len(per_image),
                      "per_image": per_image}
            results.append(result)
            print(json.dumps({key: result[key] for key in
                              ("checkpoint", "n", "means", "fraction_better_than_teacher")}),
                  flush=True)
        (output / f"{args.run_tag}_heldout_flow.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8")
        capture.close()
        return
    print(json.dumps({"train_images": len(records),
                      "trainable_parameters": sum(p.numel() for p in trainable),
                      "init_checkpoint_steps": checkpoint["steps"],
                      "train_scope": args.train_scope,
                      "early_weight": args.early_weight,
                      "anchor_weight": args.anchor_weight}), flush=True)

    def save(step):
        state = {name: value.detach().to(torch.bfloat16).cpu()
                 for name, value in model.named_parameters()
                 if value.requires_grad or
                 (args.train_scope == "revisit_only" and
                  name.startswith(f"diffusion_head_a.{args.shared_source_index}."))}
        path = output / f"{args.run_tag}_step{step}_bf16.pt"
        torch.save({"head": state, "steps": step,
                    "architecture_tuple": model._elt_architecture,
                    "source": "COCO-2017 train real-image flow matching, first-visit deep supervision, released-model anchor"}, path)
        print(json.dumps({"checkpoint": str(path)}), flush=True)

    durations = []
    model.eval()
    with metrics_path.open("w", encoding="utf-8") as stream:
        for step in range(1, args.steps + 1):
            row = rng.choice(records)
            z = load_record(row, args.data, device)
            t, noise, clean = transport.sample(z)
            t, zt, target = transport.path_sampler.plan(t, noise, clean)
            inputs = prepare_inputs(row, tokenizer, hyper,
                                    omni_attn_mask_naive,
                                    device, torch.bfloat16)
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            final, early, teacher = forward_pair(
                model, capture, teacher_layers, teacher_final,
                inputs, zt, float(t[0]), early_suffix_start)
            target = target.float()
            flow = F.mse_loss(final[:1], target)
            uncond = F.mse_loss(final[1:], target)
            early_flow = F.mse_loss(early[:1], target)
            anchor = F.mse_loss(final, teacher)
            loss = (flow + args.uncond_weight * uncond +
                    args.early_weight * early_flow +
                    args.anchor_weight * anchor)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
            if step == 1 or step % args.log_every == 0:
                result = {"step": step, "loss": round(float(loss), 5),
                          "flow": round(float(flow), 5),
                          "uncond": round(float(uncond), 5),
                          "early": round(float(early_flow), 5),
                          "anchor": round(float(anchor), 5),
                          "grad_norm": round(float(norm), 4),
                          "median_recent_s": round(statistics.median(
                              durations[-args.log_every:]), 4),
                          "peak_gb": round(torch.cuda.max_memory_allocated()/1e9, 3)}
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                print(json.dumps(result), flush=True)
            if args.save_every and step % args.save_every == 0:
                save(step)
    if not args.save_every or args.steps % args.save_every:
        save(args.steps)
    capture.close()


if __name__ == "__main__":
    main()
