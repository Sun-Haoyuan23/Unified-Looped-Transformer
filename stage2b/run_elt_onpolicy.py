"""Generate paired L1/L2 exits from the on-policy trained whole-head loop."""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import nn

from elt_model import (install_elt_head, install_residual_tail_head,
                       install_mlp_shared_tail_head, install_lora_mlp_tail_head,
                       install_svd_loop_head, install_svd_sequence_head,
                       install_svd_middle_head, install_shared_middle_head,
                       install_shared_single_head, install_svd_single_head,
                       merge_svd_adapters_for_inference,
                       load_head_checkpoint)
from early_readout import EarlyReadout, EarlyFinal, load_readout
from final_head_tuning import load_final
from loop_solver_model import install_loop_solver, load_loop_adapter
from gated_revisit import GatedRevisit, insert_gated_revisit, load_gate
from mmu_evidence_reader import MMUEvidenceReader


class GroundedFlow:
    """Use direct-latent MMU evidence to allocate tied visual visits."""

    def __init__(self, model, trained_layers, short_layers, reader, prompt,
                 check_index, threshold, correction_steps, target_index,
                 full_recompute=False):
        self.model = model
        self.full = nn.ModuleList(trained_layers)
        self.short = nn.ModuleList(short_layers)
        self.reader = reader
        self.prompt = prompt
        self.check_index = check_index
        self.threshold = threshold
        self.index = 0
        self.deep = False
        self.remaining_correction_steps = 0
        self.correction_steps = correction_steps
        self.target_index = target_index
        self.full_recompute = full_recompute
        self.full_correction_steps = 0
        self.total_block_calls = 0
        self.evidence = None
        self.evidence_check_runtime_s = 0.0

    @staticmethod
    def block_cost(layers):
        return sum(3 if isinstance(layer, GatedRevisit) else 1
                   for layer in layers)

    def continue_from_prefix(self, hidden, image_latents, t, kwargs):
        """Complete the deeper visit without rerunning the AR backbone/prefix."""
        positions = kwargs["modality_positions"]
        mask = kwargs["attention_mask"]
        tvec = t.flatten()
        if tvec.numel() == 1:
            tvec = tvec.repeat(image_latents.shape[0])
        time_embeds = self.model.time_embed(tvec, torch.bfloat16)
        position_ids = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        for layer in self.full[self.target_index:]:
            hidden = layer(hidden_states=hidden, adaln_input=time_embeds,
                           attention_mask=mask, position_ids=position_ids,
                           modality_positions=positions)[0]
        tokens = self.model.diffusion_head_b(hidden, time_embeds, positions)
        channels, height, width = image_latents.shape[1:]
        outputs = []
        for branch, (offset, length) in enumerate(positions[:, 0].tolist()):
            start = offset + int(self.model.config.add_time_embeds)
            patch_tokens = tokens[branch:branch + 1, start:offset + length]
            patches = self.model.unpatchify(
                patch_tokens, height // self.model.config.patch_size,
                width // self.model.config.patch_size)
            outputs.append(patches.transpose(1, 2).reshape(
                1, channels, height, width))
        raw = torch.cat(outputs, dim=0)
        scale = float(kwargs.get("guidance_scale", 0.0))
        if scale > 0:
            cond, uncond = torch.chunk(raw, 2, dim=0)
            guided = uncond + scale * (cond - uncond)
            return torch.cat((guided, guided), dim=0)
        return raw

    def __call__(self, image_latents, t, **kwargs):
        active = (self.full if self.remaining_correction_steps else self.short)
        if self.remaining_correction_steps:
            self.remaining_correction_steps -= 1
            self.full_correction_steps += 1
        self.model.diffusion_head_a = active
        captured = []
        hook = None
        if self.index == self.check_index and not self.full_recompute:
            hook = self.short[self.target_index - 1].register_forward_hook(
                lambda _module, _inputs, result: captured.append(result[0]))
        output = self.model.t2i_generate(image_latents, t.clone(), **kwargs)
        if hook is not None:
            hook.remove()
        self.total_block_calls += self.block_cost(active)
        if self.index == self.check_index:
            clean = image_latents[:1].float() + (1 - float(t[0])) * output[:1].float()
            started = time.perf_counter()
            self.evidence = self.reader.read(clean, self.prompt)
            self.evidence_check_runtime_s = time.perf_counter() - started
            if self.evidence["margin"] < self.threshold:
                self.deep = True
                if self.full_recompute:
                    self.model.diffusion_head_a = self.full
                    output = self.model.t2i_generate(image_latents, t.clone(), **kwargs)
                    self.total_block_calls += self.block_cost(self.full)
                else:
                    if len(captured) != 1:
                        raise RuntimeError("Expected one cached prefix state")
                    output = self.continue_from_prefix(
                        captured[0], image_latents, t, kwargs)
                    self.total_block_calls += self.block_cost(
                        self.full[self.target_index:])
                self.full_correction_steps += 1
                self.remaining_correction_steps = self.correction_steps - 1
        self.index += 1
        return output


class DepthSchedule:
    """Fixed-time 9/10-call control for a grounded-routing ablation."""

    def __init__(self, model, full_layers, short_layers, total_steps,
                 short_steps, short_first):
        self.model = model
        self.full = nn.ModuleList(full_layers)
        self.short = nn.ModuleList(short_layers)
        self.total_steps = total_steps
        self.short_steps = short_steps
        self.short_first = short_first
        self.index = 0
        self.total_block_calls = 0

    def __call__(self, image_latents, t, **kwargs):
        use_short = (self.index < self.short_steps if self.short_first else
                     self.index >= self.total_steps - self.short_steps)
        active = self.short if use_short else self.full
        self.model.diffusion_head_a = active
        output = self.model.t2i_generate(image_latents, t.clone(), **kwargs)
        self.total_block_calls += len(active)
        self.index += 1
        return output


class GuidedFlow:
    """Offline counterfactual-gradient upper-bound experiment.

    The deployed grounded loop must never use this backpropagation oracle.
    It asks whether the understanding signal can causally steer a mid-trajectory
    image at all, before attempting to amortize its effect in a shared block.
    """

    def __init__(self, model, reader, prompt, check_index, strength, window=1):
        self.model = model
        self.reader = reader
        self.prompt = prompt
        self.check_index = check_index
        self.strength = strength
        self.window = window
        self.index = 0
        self.evidence = None
        self.evidence_history = []
        self.guided_count = 0

    def __call__(self, image_latents, t, **kwargs):
        output = self.model.t2i_generate(image_latents, t.clone(), **kwargs)
        if self.check_index <= self.index < self.check_index + self.window:
            clean = (image_latents[:1].float() +
                     (1 - float(t.flatten()[0])) * output[:1].float())
            evidence = self.reader.spatial_margin_gradient(clean, self.prompt)
            gradient = evidence.pop("gradient")
            self.evidence = evidence
            self.evidence_history.append({"index": self.index, **evidence})
            if evidence["margin"] < 0:
                rms = gradient.float().square().mean().sqrt().clamp_min(1e-8)
                vrms = output[:1].float().square().mean().sqrt()
                correction = self.strength * vrms * gradient.float() / rms
                output = output + correction.to(output.dtype)
                self.evidence["gradient_rms"] = float(rms)
                self.evidence["velocity_rms"] = float(vrms)
                self.guided_count += 1
        self.index += 1
        return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--gpu-physical", type=int, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--variants", default="L1,L2")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--checkpoint", default="elt_onpolicy_5x2_bf16.pt")
    parser.add_argument("--run-tag", default="elt")
    parser.add_argument("--tasks-file", type=Path)
    parser.add_argument("--prefix-layers", type=int, default=0)
    parser.add_argument("--unique-layers", type=int, default=5)
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--architecture", choices=("original", "shared", "residual_tail",
                                                   "mlp_shared", "lora_mlp", "svd_loop", "svd_sequence", "svd_middle", "shared_middle", "shared_single", "svd_single", "loop_solver", "gated_revisit"),
                        default="shared")
    parser.add_argument("--adapter-width", type=int, default=1536)
    parser.add_argument("--solver-width", type=int, default=128)
    parser.add_argument("--gate-max-gain", type=float, default=0.1)
    parser.add_argument("--revisit-position", type=int, choices=(5, 8), default=8)
    parser.add_argument("--grounded-at-index", type=int, default=35)
    parser.add_argument("--correction-steps", type=int, default=4)
    parser.add_argument("--schedule-short-steps", type=int, default=25)
    parser.add_argument("--guide-strength", type=float, default=0.5)
    parser.add_argument("--guide-window", type=int, default=1)
    parser.add_argument("--full-recompute", action="store_true")
    parser.add_argument("--evidence-threshold", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=128)
    parser.add_argument("--svd-init", action="store_true")
    parser.add_argument("--attn-rank", type=int, default=0)
    parser.add_argument("--mlp-rank", type=int, default=0)
    parser.add_argument("--adaln-rank", type=int, default=0)
    parser.add_argument("--early-checkpoint", default="")
    parser.add_argument("--early-width", type=int, default=4096)
    parser.add_argument("--final-checkpoint", default="")
    parser.add_argument("--merged-adapters", action="store_true")
    parser.add_argument("--shared-source-index", type=int, default=5)
    parser.add_argument("--shared-target-index", type=int, default=6)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose one physical GPU by UUID")
    source = args.work / "Show-o" / "show-o2"
    sys.path.insert(0, str(source))
    from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
    from models.misc import get_text_tokenizer, prepare_gen_input
    from transport import Sampler, create_transport
    from utils import denorm, get_hyper_params

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    config = OmegaConf.load(source / "configs" / "showo2_1.5b_demo_432x432.yaml")
    tokenizer, token_ids = get_text_tokenizer(
        str(args.work / "models" / "qwen2.5-1.5b-tokenizer"),
        add_showo_tokens=True, return_showo_token_ids=True, llm_name="qwen2_5"
    )
    model = Showo2Qwen2_5.from_pretrained(
        str(args.work / "models" / "show-o2-1.5B-local"), use_safetensors=False
    ).to(device=device, dtype=dtype).eval()
    if args.architecture == "original":
        pass
    elif args.architecture == "loop_solver":
        refine = install_loop_solver(model, width=args.solver_width,
                                     trainable=False)
    elif args.architecture == "gated_revisit":
        refine = insert_gated_revisit(model, max_gain=args.gate_max_gain,
                                      position=args.revisit_position)
    elif args.architecture == "shared_middle":
        if (args.prefix_layers, args.unique_layers, args.loops) != (2, 3, 2):
            raise ValueError("Shared middle requires 2 prefix, 3 shared, 2 loops")
        install_shared_middle_head(model, trainable=False)
    elif args.architecture == "shared_single":
        install_shared_single_head(model, args.shared_source_index,
                                   args.shared_target_index, trainable=False)
    elif args.architecture == "svd_single":
        install_svd_single_head(model, args.shared_source_index,
                                args.shared_target_index, rank=args.lora_rank,
                                svd_init=args.svd_init, trainable=False)
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
    checkpoint = ({"steps": 0} if args.architecture in ("original", "shared_single", "svd_single") and
                  args.checkpoint == "none" else
                  load_loop_adapter(refine, args.work / "stage2b" / args.checkpoint)
                  if args.architecture == "loop_solver" else
                  load_gate(refine, args.work / "stage2b" / args.checkpoint)
                  if args.architecture == "gated_revisit" else
                  load_head_checkpoint(model, args.work / "stage2b" / args.checkpoint))
    if args.merged_adapters:
        merge_svd_adapters_for_inference(model)
    model.requires_grad_(False)
    if args.final_checkpoint:
        load_final(model, args.work / "stage2b" / args.final_checkpoint)
    trained_layers = list(model.diffusion_head_a)
    trained_final = model.diffusion_head_b
    early_final = None
    if args.early_checkpoint:
        readout = EarlyReadout(model.diffusion_head_config.hidden_size,
                               args.early_width).to(device)
        load_readout(readout, args.work / "stage2b" / args.early_checkpoint)
        early_final = EarlyFinal(trained_final, readout.eval())
    vae = WanVAE(vae_pth="/home/shy_23/models/show-o2-deps/Wan2.1_VAE.pth",
                 dtype=dtype, device=device)

    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1
    (nvisual, _nmmu, _nvideo, max_seq_len, max_text_len, channels, patch_size,
     latent_w, latent_h, pad_id, bos_id, eos_id, boi_id, eoi_id,
     _bov_id, _eov_id, img_pad_id, _vid_pad_id, _guidance) = get_hyper_params(
        config, tokenizer, token_ids
    )
    transport = create_transport(
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight,
        train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps,
        snr_type=config.transport.snr_type,
        do_shift=config.transport.do_shift,
        seq_len=nvisual,
    )
    sample_fn = Sampler(transport).sample_ode(
        sampling_method=config.transport.sampling_method,
        num_steps=args.steps,
        atol=config.transport.atol,
        rtol=config.transport.rtol,
        reverse=config.transport.reverse,
        time_shifting_factor=config.transport.time_shifting_factor,
    )
    tasks = json.loads((args.tasks_file or args.work / "prompts.json").read_text(
        encoding="utf-8"))
    tasks = [task for i, task in enumerate(tasks) if i % args.num_shards == args.shard]
    jobs = [(task, seed) for task in tasks for seed in task["seeds"]]
    if args.max_images:
        jobs = jobs[:args.max_images]
    variants = args.variants.split(",")
    if any(v not in ("L1", "L2", "G", "E", "T", "U") for v in variants):
        raise ValueError(variants)
    if "U" in variants and args.architecture != "original":
        raise ValueError("Gradient oracle uses the released original model")
    if "U" in variants and not 1 <= args.guide_window <= args.steps - 1 - args.grounded_at_index:
        raise ValueError("Gradient oracle window outside flow evaluations")
    if any(v in ("E", "T") for v in variants) and args.architecture not in ("shared_single", "svd_single"):
        raise ValueError("Time-routing ablation requires a single shared visit")
    if not 0 <= args.schedule_short_steps <= args.steps - 1:
        raise ValueError("Invalid number of short-schedule steps")
    if "G" in variants and args.architecture not in ("shared_middle", "shared_single", "svd_single", "gated_revisit"):
        raise ValueError("Grounded variant requires a shared loop")
    if "G" in variants and args.correction_steps < 1:
        raise ValueError("Correction window must contain at least one step")
    if "G" in variants and not 0 <= args.grounded_at_index < args.steps - 1:
        raise ValueError("Grounding index outside flow evaluations")
    reader = (MMUEvidenceReader(model, tokenizer, token_ids, _nmmu, device)
              if "G" in variants or "U" in variants else None)
    root = args.work / "stage2b"
    image_dir = root / f"{args.run_tag}_images"
    log_dir = root / "logs"
    image_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / f"{args.run_tag}_metrics_gpu{args.gpu_physical}.jsonl"
    print(json.dumps({"gpu": args.gpu_physical, "jobs": len(jobs),
                      "variants": variants, "checkpoint_steps": checkpoint["steps"]}),
          flush=True)
    with metrics_path.open("w", encoding="utf-8") as metrics:
        for task, seed in jobs:
            (cond, uncond, cond_pos, uncond_pos) = prepare_gen_input(
                [task["prompt"]], tokenizer, nvisual, bos_id, eos_id,
                boi_id, eoi_id, pad_id, img_pad_id, max_text_len, device
            )
            tokens = torch.cat((cond, uncond), dim=0)
            positions = torch.cat((cond_pos, uncond_pos), dim=0)
            mask = omni_attn_mask_naive(tokens.size(0), max_seq_len,
                                        positions, device).to(dtype)
            kwargs = dict(text_tokens=tokens, attention_mask=mask,
                          modality_positions=positions, output_hidden_states=True,
                          max_seq_len=max_seq_len, guidance_scale=args.guidance)
            generator = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(
                (1, channels, latent_h * patch_size, latent_w * patch_size),
                generator=generator, device=device, dtype=dtype
            )
            z = torch.cat((noise, noise), dim=0)
            for variant in variants:
                depth = 1 if variant in ("G", "E", "T", "U") else int(variant[-1])
                call_count = (10 if args.architecture == "original" else
                              9 if args.architecture in ("shared_single", "svd_single") and depth == 1 else
                              10 if args.architecture in ("shared_single", "svd_single") else
                              7 if args.architecture == "shared_middle" and depth == 1 else
                              10 if args.architecture == "shared_middle" else
                              13 if args.architecture == "gated_revisit" else
                              10 * depth if args.architecture == "loop_solver" else
                              args.prefix_layers + args.unique_layers * depth)
                if args.architecture == "gated_revisit" and depth != 2 and variant != "G":
                    raise ValueError("Gated revisit pilot only supports L2")
                if args.architecture == "original":
                    active_layers = trained_layers
                elif args.architecture == "shared_middle" and depth == 1:
                    active_layers = trained_layers[:5] + trained_layers[-2:]
                elif args.architecture in ("shared_single", "svd_single") and depth == 1:
                    i = args.shared_target_index
                    active_layers = trained_layers[:i] + trained_layers[i+1:]
                else:
                    active_layers = trained_layers[:(
                        11 if args.architecture == "gated_revisit" else call_count)]
                model.diffusion_head_a = nn.ModuleList(active_layers)
                model.diffusion_head_b = early_final if depth == 1 and early_final else trained_final
                record = {"id": task["id"], "category": task["category"],
                          "prompt": task["prompt"], "seed": seed,
                          "variant": variant,
                          "head_block_calls_per_velocity": (None if variant in ("G", "E", "T", "U") else call_count),
                          "flow_velocity_calls": args.steps - 1,
                          "gpu_physical": args.gpu_physical, "guidance": args.guidance,
                          "steps": args.steps, "checkpoint_steps": checkpoint["steps"]}
                try:
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        if variant == "G":
                            if args.architecture in ("shared_single", "svd_single"):
                                i = args.shared_target_index
                                short_layers = trained_layers[:i] + trained_layers[i+1:]
                                target_index = i
                            elif args.architecture == "gated_revisit":
                                i = args.revisit_position
                                short_layers = trained_layers[:i] + trained_layers[i+1:]
                                target_index = i
                            else:
                                short_layers = trained_layers[:5] + trained_layers[-2:]
                                target_index = 5
                            controller = GroundedFlow(
                                model, trained_layers, short_layers, reader,
                                task["prompt"], args.grounded_at_index,
                                args.evidence_threshold, args.correction_steps,
                                target_index, args.full_recompute)
                            latents = sample_fn(z.clone(), controller, **kwargs)[-1]
                            record.update({"total_visual_block_calls": controller.total_block_calls,
                                           "extra_loop_triggered": controller.deep,
                                           "evidence": controller.evidence,
                                           "evidence_check_index": args.grounded_at_index,
                                           "evidence_check_runtime_s": controller.evidence_check_runtime_s,
                                           "full_correction_steps": controller.full_correction_steps})
                        elif variant == "U":
                            controller = GuidedFlow(
                                model, reader, task["prompt"],
                                args.grounded_at_index, args.guide_strength,
                                args.guide_window)
                            latents = sample_fn(z.clone(), controller, **kwargs)[-1]
                            record.update({"total_visual_block_calls":
                                           call_count * (args.steps - 1),
                                           "evidence": controller.evidence,
                                            "gradient_oracle_applied": controller.guided_count > 0,
                                            "gradient_oracle_count": controller.guided_count,
                                            "evidence_history": controller.evidence_history,
                                           "guide_strength": args.guide_strength,
                                            "guide_window": args.guide_window,
                                           "evidence_check_index": args.grounded_at_index})
                        elif variant in ("E", "T"):
                            i = args.shared_target_index
                            short_layers = trained_layers[:i] + trained_layers[i+1:]
                            controller = DepthSchedule(
                                model, trained_layers, short_layers, args.steps - 1,
                                args.schedule_short_steps, variant == "E")
                            latents = sample_fn(z.clone(), controller, **kwargs)[-1]
                            record["total_visual_block_calls"] = controller.total_block_calls
                            record["schedule_short_steps"] = args.schedule_short_steps
                        else:
                            latents = sample_fn(z.clone(), model.t2i_generate, **kwargs)[-1]
                            record["total_visual_block_calls"] = call_count * (args.steps - 1)
                    torch.cuda.synchronize()
                    record["denoise_s"] = round(time.perf_counter() - started, 4)
                    latents = torch.chunk(latents, 2, dim=0)[0]
                    with torch.no_grad():
                        pixels = vae.batch_decode(latents.unsqueeze(2)).squeeze(2)
                    image_path = image_dir / f"{task['id']}_seed{seed}_{variant}.png"
                    Image.fromarray(denorm(pixels)[0]).save(image_path)
                    torch.cuda.synchronize()
                    record.update({"status": "ok", "image_path": str(image_path),
                                   "total_s": round(time.perf_counter() - started, 4),
                                   "peak_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 3)})
                except Exception as error:
                    record.update({"status": "error", "error": str(error),
                                   "traceback": traceback.format_exc(limit=6)})
                metrics.write(json.dumps(record, ensure_ascii=False) + "\n")
                metrics.flush()
                print(json.dumps({k: record.get(k) for k in
                                  ("id", "seed", "variant", "status", "total_s", "error")}),
                      flush=True)


if __name__ == "__main__":
    main()
