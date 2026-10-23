"""Paired, shared-weight visual-token loop probe for released Show-o2 1.5B.

This is an untrained intervention, not the proposed learned observer/feedbacker.
It repeats the full pretrained flow head at selected *actual flow times*, blends
the second-pass hidden state into visual tokens only, and keeps all other
sampling settings fixed. Every variant starts from the identical seeded noise.
"""

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


class VisualLoop:
    def __init__(self, model, targets, depth, alpha, cond_only=True):
        self.model = model
        self.targets = targets
        self.depth = depth
        self.alpha = alpha
        self.cond_only = cond_only
        self.active = False
        self.nested = False
        self.done = set()
        self.events = []
        self.handle = model.diffusion_head_a[-1].register_forward_hook(
            self._hook, with_kwargs=True
        )

    def close(self):
        self.handle.remove()

    def predict(self, image_latents, t, **kwargs):
        value = float(t.flatten()[0].item())
        selected = None
        for target in self.targets:
            if target not in self.done and value >= target:
                selected = target
                self.done.add(target)
                break
        self.active = selected is not None
        if self.active:
            self.events.append({"target_t": selected, "actual_t": value,
                                "latent_rms": float(image_latents.float().square().mean().sqrt())})
        try:
            return self.model.t2i_generate(image_latents=image_latents, t=t, **kwargs)
        finally:
            self.active = False

    def _hook(self, module, args, kwargs, output):
        if not self.active or self.nested:
            return output
        first = output[0]
        positions = kwargs["modality_positions"]
        mask = torch.zeros(first.shape[:2], device=first.device, dtype=torch.bool)
        batch_count = first.shape[0] // 2 if self.cond_only else first.shape[0]
        for batch in range(batch_count):
            for offset, length in positions[batch].tolist():
                if length > 1:
                    start = offset + int(self.model.config.add_time_embeds)
                    mask[batch, start:offset + length] = True
        self.nested = True
        try:
            current = first
            for _ in range(self.depth):
                for layer in self.model.diffusion_head_a:
                    current = layer(hidden_states=current,
                                    adaln_input=kwargs["adaln_input"],
                                    attention_mask=kwargs["attention_mask"],
                                    position_ids=kwargs["position_ids"],
                                    modality_positions=positions)[0]
        finally:
            self.nested = False
        correction = (current - first) * self.alpha
        result = first + torch.where(mask.unsqueeze(-1), correction, 0)
        self.events[-1]["visual_delta_rms"] = float(
            correction[mask].float().square().mean().sqrt()
        )
        self.events[-1]["visual_tokens"] = int(mask.sum().item())
        return (result,) + tuple(output[1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--gpu-physical", type=int, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--variants", default="none,early,mid,late,all")
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose one physical GPU per worker by UUID")

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
    config.model.showo.llm_vocab_size = len(tokenizer)
    model = Showo2Qwen2_5.from_pretrained(
        str(args.work / "models" / "show-o2-1.5B-local"), use_safetensors=False
    ).to(device=device, dtype=dtype).eval()
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("Model has meta parameters")
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

    tasks = json.loads((args.work / "prompts.json").read_text(encoding="utf-8"))
    if args.max_prompts:
        tasks = tasks[:args.max_prompts]
    tasks = [task for i, task in enumerate(tasks) if i % args.num_shards == args.shard]
    jobs = [(task, seed) for task in tasks for seed in task["seeds"]]
    if args.max_images:
        jobs = jobs[:args.max_images]
    variants = args.variants.split(",")
    target_map = {"none": [], "early": [0.25], "mid": [0.5],
                  "late": [0.75], "all": [0.25, 0.5, 0.75]}
    if any(v not in target_map for v in variants):
        raise ValueError(f"Unknown variant in {variants}")
    result_dir = args.work / "stage2_loop"
    image_dir = result_dir / "images"
    log_dir = result_dir / "logs"
    image_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    metric_path = log_dir / f"metrics_gpu{args.gpu_physical}.jsonl"
    print(json.dumps({"gpu": args.gpu_physical, "jobs": len(jobs),
                      "variants": variants, "alpha": args.alpha,
                      "depth": args.depth}), flush=True)
    with metric_path.open("w", encoding="utf-8") as metrics:
        for task, seed in jobs:
            (cond, uncond, cond_pos, uncond_pos) = prepare_gen_input(
                [task["prompt"]], tokenizer, nvisual, bos_id, eos_id,
                boi_id, eoi_id, pad_id, img_pad_id, max_text_len, device
            )
            tokens = torch.cat((cond, uncond), dim=0)
            positions = torch.cat((cond_pos, uncond_pos), dim=0)
            mask = omni_attn_mask_naive(
                tokens.size(0), max_seq_len, positions, device
            ).to(dtype)
            kwargs = dict(text_tokens=tokens, attention_mask=mask,
                          modality_positions=positions, output_hidden_states=True,
                          max_seq_len=max_seq_len, guidance_scale=args.guidance)
            noise_rng = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(
                (1, channels, latent_h * patch_size, latent_w * patch_size),
                generator=noise_rng, device=device, dtype=dtype
            )
            z = torch.cat((noise, noise), dim=0)
            for variant in variants:
                record = {"id": task["id"], "category": task["category"],
                          "prompt": task["prompt"], "seed": seed,
                          "gpu_physical": args.gpu_physical,
                          "variant": variant, "steps": args.steps,
                          "guidance": args.guidance, "depth": args.depth,
                          "alpha": args.alpha, "targets": target_map[variant]}
                controller = None
                try:
                    torch.cuda.reset_peak_memory_stats(device)
                    controller = VisualLoop(model, target_map[variant],
                                            args.depth, args.alpha)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with torch.no_grad():
                        latents = sample_fn(z.clone(), controller.predict, **kwargs)[-1]
                    torch.cuda.synchronize()
                    record["denoise_s"] = round(time.perf_counter() - started, 4)
                    latents = torch.chunk(latents, 2, dim=0)[0]
                    with torch.no_grad():
                        pixels = vae.batch_decode(latents.unsqueeze(2)).squeeze(2)
                    image = Image.fromarray(denorm(pixels)[0])
                    image_path = image_dir / f"{task['id']}_seed{seed}_{variant}.png"
                    image.save(image_path)
                    torch.cuda.synchronize()
                    record.update({"status": "ok", "image_path": str(image_path),
                                   "total_s": round(time.perf_counter() - started, 4),
                                   "events": controller.events,
                                   "peak_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 3)})
                    if len(controller.events) != len(target_map[variant]):
                        raise RuntimeError("Not all target flow times were reached")
                except Exception as error:
                    record.update({"status": "error", "error": str(error),
                                   "traceback": traceback.format_exc(limit=6)})
                finally:
                    if controller is not None:
                        controller.close()
                    metrics.write(json.dumps(record, ensure_ascii=False) + "\n")
                    metrics.flush()
                    print(json.dumps({k: record.get(k) for k in
                                      ("id", "seed", "variant", "status", "total_s", "error")}),
                          flush=True)


if __name__ == "__main__":
    main()
