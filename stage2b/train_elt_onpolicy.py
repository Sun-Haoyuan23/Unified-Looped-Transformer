"""Train a 5N x 2L whole-head loop on real teacher ODE trajectories.

This is ELT-style ILSD adapted to a pretrained hybrid flow model. The external
Show-o2 teacher velocity bootstraps the ground-truth term. The 2-loop output
is the online teacher for the 1-loop prefix, with lambda annealed 1 -> 0.
"""

import argparse
import copy
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.nn import functional as F

from elt_model import (install_elt_head, install_residual_tail_head,
                       install_mlp_shared_tail_head, install_lora_mlp_tail_head,
                       install_svd_loop_head, install_svd_sequence_head,
                       install_svd_middle_head, install_shared_middle_head,
                       install_shared_single_head, install_svd_single_head,
                       load_head_checkpoint)


class FirstLoopCapture:
    def __init__(self, model, first_exit_index, extra_index=None):
        self.states = []
        self.first_exit_index = first_exit_index
        self.extra_index = extra_index
        self.handle = model.diffusion_head_a[first_exit_index].register_forward_hook(self.capture)
        self.extra_handle = (model.diffusion_head_a[extra_index].register_forward_hook(self.capture)
                             if extra_index is not None else None)
        self.second_handle = None
        if model.diffusion_head_a[first_exit_index] is not model.diffusion_head_a[-1]:
            self.second_handle = model.diffusion_head_a[-1].register_forward_hook(self.capture)
        self.initial_state = None
        self.input_handle = model.diffusion_head_a[0].register_forward_pre_hook(
            self.capture_input, with_kwargs=True)

    def capture_input(self, module, args, kwargs):
        if self.initial_state is None:
            self.initial_state = kwargs["hidden_states"].detach()

    def capture(self, module, inputs, output):
        self.states.append(output[0])

    def close(self):
        self.handle.remove()
        if self.extra_handle is not None:
            self.extra_handle.remove()
        if self.second_handle is not None:
            self.second_handle.remove()
        self.input_handle.remove()


def visual_representation_loss(student, teacher, positions, eps=1e-6):
    losses = []
    for s, t in zip(student, teacher):
        for branch, (offset, length) in enumerate(positions[:, 0].tolist()):
            sv = s[branch, offset:offset + length].float()
            tv = t[branch, offset:offset + length].float()
            # Normalize by teacher feature energy so this scale is stable
            # across early and late blocks and across diffusion time.
            losses.append(F.mse_loss(sv, tv) / tv.square().mean().clamp_min(eps))
    return torch.stack(losses).mean()


def to_velocity(model, hidden, time_embeds, positions, channels, height, width,
                early=False):
    if early and hasattr(model, "_elt_early_readout"):
        hidden = model._elt_early_readout(hidden)
    token_output = model.diffusion_head_b(hidden, time_embeds, positions)
    # CFG branches have different image offsets because the unconditional text
    # is empty. Match Show-o2.forward's per-branch unpatchify exactly.
    pixels = []
    for branch, (offset, length) in enumerate(positions[:, 0].tolist()):
        start = offset + int(model.config.add_time_embeds)
        tokens = token_output[branch:branch + 1, start:offset + length]
        branch_pixels = model.unpatchify(
            tokens, height // model.config.patch_size,
            width // model.config.patch_size)
        pixels.append(branch_pixels.transpose(1, 2).reshape(
            1, channels, height, width))
    return torch.cat(pixels, dim=0)


def teacher_head_outputs(model, capture, teacher_layers, teacher_final,
                         inputs, zt, t):
    _tokens, positions, mask = inputs
    hidden = capture.initial_state
    if hidden is None:
        raise RuntimeError("Missing frozen-backbone input to visual head")
    time_embeds = model.time_embed(
        torch.full((2,), float(t), device=zt.device, dtype=torch.float32),
        torch.bfloat16)
    position_ids = torch.arange(hidden.shape[1], device=zt.device).unsqueeze(0)
    states = []
    with torch.no_grad():
        for index, layer in enumerate(teacher_layers):
            hidden = layer(hidden_states=hidden, adaln_input=time_embeds,
                           attention_mask=mask, position_ids=position_ids,
                           modality_positions=positions)[0]
            if index in (capture.first_exit_index, capture.extra_index, 9):
                states.append(hidden.detach())
        # Temporarily use the original shared prediction head for the exact
        # pretrained branch velocities; no duplicate Qwen/semantic pass.
        token_output = teacher_final(hidden, time_embeds, positions)
        pixels = []
        channels, height, width = zt.shape[1:]
        for branch, (offset, length) in enumerate(positions[:, 0].tolist()):
            start = offset + int(model.config.add_time_embeds)
            tokens = token_output[branch:branch + 1, start:offset + length]
            branch_pixels = model.unpatchify(
                tokens, height // model.config.patch_size,
                width // model.config.patch_size)
            pixels.append(branch_pixels.transpose(1, 2).reshape(
                1, channels, height, width))
    return torch.cat(pixels, dim=0), states


def guide(raw, scale):
    cond, uncond = torch.chunk(raw, 2, dim=0)
    return uncond + scale * (cond - uncond)


def prepare_inputs(job, tokenizer, hyper, mask_fn, device, dtype):
    from models.misc import prepare_gen_input
    (nvisual, max_seq_len, max_text_len, pad_id, bos_id, eos_id,
     boi_id, eoi_id, img_pad_id) = hyper
    cond, uncond, cpos, upos = prepare_gen_input(
        [job["prompt"]], tokenizer, nvisual, bos_id, eos_id,
        boi_id, eoi_id, pad_id, img_pad_id, max_text_len, device)
    tokens = torch.cat((cond, uncond), dim=0)
    positions = torch.cat((cpos, upos), dim=0)
    mask = mask_fn(2, max_seq_len, positions, device).to(dtype)
    return tokens, positions, mask


def forward_exits(model, capture, input_data, zt, t, scale,
                  return_raw=False):
    tokens, positions, mask = input_data
    capture.states.clear()
    capture.initial_state = None
    zt = zt.to(model.device if hasattr(model, "device") else tokens.device)
    zt = torch.cat((zt, zt), dim=0)
    t = torch.full((2,), float(t), device=zt.device, dtype=torch.float32)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _logits, raw_final = model(
            text_tokens=tokens, image_latents=zt, t=t,
            attention_mask=mask, modality_positions=positions,
            output_hidden_states=True, max_seq_len=mask.shape[-1], device=zt.device)
        shared_middle = model._elt_architecture[0] == "shared_middle"
        shared_single = model._elt_architecture[0] in ("shared_single", "svd_single")
        expected = 3 if shared_middle or shared_single else 2
        if len(capture.states) != expected:
            raise RuntimeError(f"Expected {expected} captured states; got {len(capture.states)}")
        time_embeds = model.time_embed(t, torch.bfloat16)
        channels, height, width = zt.shape[1:]
        first_state = capture.states[0]
        if shared_middle:
            position_ids = torch.arange(first_state.shape[1], device=zt.device).unsqueeze(0)
            for layer in model.diffusion_head_a[-2:]:
                first_state = layer(hidden_states=first_state,
                                    adaln_input=time_embeds,
                                    attention_mask=mask,
                                    position_ids=position_ids,
                                    modality_positions=positions)[0]
        elif shared_single:
            position_ids = torch.arange(first_state.shape[1], device=zt.device).unsqueeze(0)
            target_index = model._elt_architecture[2]
            for layer in model.diffusion_head_a[target_index + 1:]:
                first_state = layer(hidden_states=first_state,
                                    adaln_input=time_embeds,
                                    attention_mask=mask,
                                    position_ids=position_ids,
                                    modality_positions=positions)[0]
        raw_first = to_velocity(model, first_state, time_embeds, positions,
                                channels, height, width, early=True)
        if not getattr(capture, "verified", False):
            reconstructed = to_velocity(
                model, capture.states[expected - 1], time_embeds, positions,
                channels, height, width)
            torch.testing.assert_close(reconstructed, raw_final,
                                       rtol=1e-3, atol=1e-3)
            capture.verified = True
    first, final = guide(raw_first.float(), scale), guide(raw_final.float(), scale)
    if return_raw:
        return first, final, raw_first.float(), raw_final.float()
    return first, final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpu-physical", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--init", choices=("first", "last", "pair_average"),
                        default="first")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--trajectory-dirs", default="train")
    parser.add_argument("--lambda-start", type=float, default=1.0)
    parser.add_argument("--lambda-end", type=float, default=0.0)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--checkpoint-precision", choices=("bf16", "fp32"),
                        default="bf16")
    parser.add_argument("--representation-weight", type=float, default=0.0)
    parser.add_argument("--raw-weight", type=float, default=0.0)
    parser.add_argument("--shared-source-index", type=int, default=1)
    parser.add_argument("--shared-target-index", type=int, default=2)
    parser.add_argument("--prefix-layers", type=int, default=0)
    parser.add_argument("--unique-layers", type=int, default=5)
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--architecture", choices=("shared", "residual_tail",
                                                   "mlp_shared", "lora_mlp", "svd_loop", "svd_sequence", "svd_middle", "shared_middle", "shared_single", "svd_single"),
                        default="shared")
    parser.add_argument("--adapter-width", type=int, default=1536)
    parser.add_argument("--lora-rank", type=int, default=128)
    parser.add_argument("--attn-rank", type=int, default=0)
    parser.add_argument("--mlp-rank", type=int, default=0)
    parser.add_argument("--adaln-rank", type=int, default=0)
    parser.add_argument("--svd-init", action="store_true")
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose one healthy GPU by UUID")
    torch.set_float32_matmul_precision("high")
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
    teacher_layers = None
    teacher_final = None
    if args.representation_weight or args.raw_weight:
        teacher_layers = copy.deepcopy(model.diffusion_head_a).eval()
        teacher_final = copy.deepcopy(model.diffusion_head_b).eval()
        for parameter in list(teacher_layers.parameters()) + list(teacher_final.parameters()):
            parameter.requires_grad_(False)
    if args.architecture == "residual_tail":
        trainable = install_residual_tail_head(
            model, prefix_layers=args.prefix_layers,
            unique_layers=args.unique_layers,
            adapter_width=args.adapter_width, trainable=True)
    elif args.architecture == "mlp_shared":
        trainable = install_mlp_shared_tail_head(
            model, prefix_layers=args.prefix_layers,
            unique_layers=args.unique_layers, trainable=True)
    elif args.architecture == "lora_mlp":
        trainable = install_lora_mlp_tail_head(
            model, prefix_layers=args.prefix_layers,
            unique_layers=args.unique_layers, rank=args.lora_rank, trainable=True)
    elif args.architecture in ("svd_loop", "svd_sequence", "svd_middle"):
        if (args.prefix_layers, args.unique_layers, args.loops) != (0, 5, 2):
            raise ValueError("SVD layouts use ten executed blocks; leave depth flags at defaults")
        installer = {"svd_loop": install_svd_loop_head,
                     "svd_sequence": install_svd_sequence_head,
                     "svd_middle": install_svd_middle_head}[args.architecture]
        trainable = installer(
            model, rank=args.lora_rank, svd_init=args.svd_init, trainable=True,
            attention_rank=args.attn_rank, mlp_rank=args.mlp_rank,
            adaln_rank=args.adaln_rank)
    elif args.architecture == "shared_middle":
        if (args.prefix_layers, args.unique_layers, args.loops) != (2, 3, 2):
            raise ValueError("Shared middle requires --prefix-layers 2 --unique-layers 3 --loops 2")
        trainable = install_shared_middle_head(model, trainable=True)
    elif args.architecture == "shared_single":
        trainable = install_shared_single_head(
            model, args.shared_source_index, args.shared_target_index,
            trainable=True)
    elif args.architecture == "svd_single":
        trainable = install_svd_single_head(
            model, args.shared_source_index, args.shared_target_index,
            rank=args.lora_rank, svd_init=args.svd_init, trainable=True)
    else:
        trainable = install_elt_head(model, unique_layers=args.unique_layers,
                                     loops=args.loops,
                                     precision=torch.float32, trainable=True,
                                     init=args.init, prefix_layers=args.prefix_layers)
    if args.resume:
        load_head_checkpoint(model, work / "stage2b" / args.resume)
    first_exit_index = (args.shared_target_index - 1
                        if args.architecture in ("shared_single", "svd_single") else
                        args.prefix_layers + args.unique_layers - 1)
    capture = FirstLoopCapture(
        model, first_exit_index,
        extra_index=(args.shared_target_index
                     if args.architecture in ("shared_single", "svd_single") else None))
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.99),
                                  weight_decay=0.01)
    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1
    (nvisual, _nmmu, _nvideo, max_seq_len, max_text_len, _channels, _patch,
     _w, _h, pad_id, bos_id, eos_id, boi_id, eoi_id,
     _bov, _eov, img_pad_id, _vid, _guide) = get_hyper_params(config, tokenizer, token_ids)
    hyper = (nvisual, max_seq_len, max_text_len, pad_id, bos_id, eos_id,
             boi_id, eoi_id, img_pad_id)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = manifest["train"]
    if args.max_records:
        jobs = jobs[:args.max_records]
    records = []
    data_sources = [item.strip() for item in args.trajectory_dirs.split(",")]
    for job in jobs:
        input_data = prepare_inputs(job, tokenizer, hyper, omni_attn_mask_naive,
                                    device, dtype)
        for directory in data_sources:
            file = work / "stage2b" / "trajectories" / directory / f"{job['id']}.pt"
            if not file.exists():
                if args.allow_missing:
                    continue
                raise FileNotFoundError(file)
            try:
                payload = torch.load(file, map_location="cpu", weights_only=False)
            except Exception:
                if args.allow_missing:
                    print(f"Skipping unfinished trajectory: {file}", flush=True)
                    continue
                raise
            if payload["prompt"] != job["prompt"] or payload["zt"].shape[0] != 49:
                raise RuntimeError(f"Invalid trajectory: {file}")
            records.append((payload, input_data))
    rng = random.Random(args.seed)
    print(json.dumps({"train_prompts": len(jobs), "trajectory_states": len(records)*49,
                      "trainable_parameters": sum(p.numel() for p in trainable),
                      "data_sources": data_sources, "resume": args.resume,
                      "representation_weight": args.representation_weight,
                      "raw_weight": args.raw_weight,
                      "architecture": f"{args.architecture}: {args.prefix_layers} prefix + {args.unique_layers} shared x {args.loops} loops",
                      "max_head_calls": 10}), flush=True)
    # Trainable parameters still receive gradients in eval mode. Keep the
    # pretrained hybrid backbone's inference behaviour exactly unchanged.
    model.eval()
    output = work / "stage2b"
    output.mkdir(parents=True, exist_ok=True)
    tag = args.run_tag or ("elt_onpolicy_5x2" if args.init == "first"
                           else f"elt_onpolicy_5x2_{args.init}")
    metrics = output / f"train_{tag}.jsonl"
    def save_head(path, completed_steps):
        saved_dtype = (torch.float32 if args.checkpoint_precision == "fp32"
                       else torch.bfloat16)
        state = {name: p.detach().to(saved_dtype).cpu()
                 for name, p in model.named_parameters() if p.requires_grad}
        torch.save({"head": state, "steps": completed_steps,
                    "architecture": f"{args.architecture}: {args.prefix_layers} prefix + {args.unique_layers} shared x {args.loops} loops",
                    "architecture_tuple": model._elt_architecture,
                    "source": "pretrained Show-o2 teacher on-policy ODE velocity; ILSD",
                    "init": args.init, "resume": args.resume,
                    "data_sources": data_sources}, path)
    recent, step_times = [], []
    torch.cuda.reset_peak_memory_stats(device)
    with metrics.open("w", encoding="utf-8") as stream:
        for step in range(args.steps):
            payload, inputs = records[rng.randrange(len(records))]
            index = rng.randrange(49)
            zt = payload["zt"][index].to(device)
            t = payload["t"][index]
            target = payload["teacher_velocity"][index].to(device).float()
            scale = float(payload["guidance"])
            lam = (args.lambda_start +
                   (args.lambda_end - args.lambda_start) *
                   step / max(args.steps - 1, 1))
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            if teacher_layers is not None:
                first, final, raw_first, raw_final = forward_exits(
                    model, capture, inputs, zt, t, scale, return_raw=True)
            else:
                first, final = forward_exits(model, capture, inputs, zt, t, scale)
            loss_max = F.mse_loss(final, target)
            loss_early = F.mse_loss(first, target)
            loss_ilsd = F.mse_loss(first, final.detach())
            loss = (loss_max if args.architecture in ("residual_tail", "lora_mlp", "svd_loop", "svd_sequence", "svd_middle") else
                    loss_max + lam * loss_early + (1 - lam) * loss_ilsd)
            loss_representation = None
            loss_raw = None
            if teacher_layers is not None:
                teacher_raw, teacher_states = teacher_head_outputs(
                    model, capture, teacher_layers, teacher_final,
                    inputs, zt, t)
                positions = inputs[1]
                if args.representation_weight:
                    selected_states = ([capture.states[0], capture.states[2]]
                                       if args.architecture == "shared_middle" else capture.states)
                    loss_representation = visual_representation_loss(
                        selected_states, teacher_states, positions)
                    loss = loss + args.representation_weight * loss_representation
                if args.raw_weight:
                    target_raw = teacher_raw.float()
                    loss_raw = (F.mse_loss(raw_final, target_raw) +
                                lam * F.mse_loss(raw_first, target_raw) +
                                (1 - lam) * F.mse_loss(raw_first,
                                                       raw_final.detach()))
                    loss = loss + args.raw_weight * loss_raw
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step+1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            recent.append(float(loss.detach()))
            step_times.append(elapsed)
            if (step + 1) % args.log_every == 0 or step == 0:
                row = {"step": step+1, "loss": round(float(loss), 6),
                       "loss_max": round(float(loss_max), 6),
                       "loss_early": round(float(loss_early), 6),
                       "loss_ilsd": round(float(loss_ilsd), 6),
                       "loss_representation": (round(float(loss_representation), 6)
                                               if loss_representation is not None else None),
                       "loss_raw": (round(float(loss_raw), 6)
                                    if loss_raw is not None else None),
                       "lambda": round(lam, 5),
                       "median_recent_s": round(statistics.median(step_times[-args.log_every:]), 4),
                       "peak_allocated_gb": round(torch.cuda.max_memory_allocated(device)/1e9, 3)}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(json.dumps(row), flush=True)
            if args.save_every and not args.no_save and (step + 1) % args.save_every == 0:
                intermediate = output / f"{tag}_step{step+1}_{args.checkpoint_precision}.pt"
                save_head(intermediate, step + 1)
                print(json.dumps({"checkpoint": str(intermediate),
                                  "step": step + 1}), flush=True)
    checkpoint = output / f"{tag}_{args.checkpoint_precision}.pt"
    if not args.no_save:
        save_head(checkpoint, args.steps)
    summary = {"steps": args.steps, "train_prompts": len(jobs),
               "trajectory_states": len(records)*49,
               "data_sources": data_sources, "resume": args.resume,
               "trainable_parameters": sum(p.numel() for p in trainable),
               "median_step_s_after_warmup": round(statistics.median(step_times[min(20,len(step_times)//4):]), 4),
               "peak_allocated_gb": round(torch.cuda.max_memory_allocated(device)/1e9, 3),
               "checkpoint": str(checkpoint)}
    (output / f"train_summary_{tag}.json").write_text(
        json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary), flush=True)
    capture.close()


if __name__ == "__main__":
    main()
