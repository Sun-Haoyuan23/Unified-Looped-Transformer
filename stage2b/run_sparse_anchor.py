"""Test sparse original-head anchors while using the trained L1 loop elsewhere."""

import argparse
import copy
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

from elt_model import install_elt_head, load_head_checkpoint


def anchor_steps(schedule):
    if schedule == "student_L1":
        return set()
    if schedule == "ends5":
        return set(range(5)) | set(range(44, 49))
    if schedule == "ends10":
        return set(range(10)) | set(range(39, 49))
    if schedule == "periodic5":
        return set(range(0, 49, 5))
    if schedule == "teacher":
        return set(range(49))
    raise ValueError(schedule)


class SparseAnchor:
    def __init__(self, model, teacher_layers, teacher_final,
                 student_layers, student_final, schedule):
        self.model = model
        self.teacher_layers = teacher_layers
        self.teacher_final = teacher_final
        self.student_layers = student_layers
        self.student_final = student_final
        self.anchors = anchor_steps(schedule)
        self.index = 0

    def __call__(self, image_latents, t, **kwargs):
        if self.index in self.anchors:
            self.model.diffusion_head_a = self.teacher_layers
            self.model.diffusion_head_b = self.teacher_final
        else:
            self.model.diffusion_head_a = self.student_layers
            self.model.diffusion_head_b = self.student_final
        self.index += 1
        return self.model.t2i_generate(image_latents, t, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--gpu-physical", type=int, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--variants", default="student_L1,ends5,ends10,periodic5,teacher")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--checkpoint", default="elt_onpolicy_5x2_bf16.pt")
    parser.add_argument("--run-tag", default="anchor")
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
    teacher_layers = copy.deepcopy(model.diffusion_head_a)
    teacher_final = copy.deepcopy(model.diffusion_head_b)
    install_elt_head(model, unique_layers=5, loops=2,
                     precision=torch.float32, trainable=False)
    checkpoint = load_head_checkpoint(
        model, args.work / "stage2b" / args.checkpoint)
    student_layers = nn.ModuleList(list(model.diffusion_head_a)[:5])
    student_final = model.diffusion_head_b
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
    tasks = [task for i, task in enumerate(tasks) if i % args.num_shards == args.shard]
    jobs = [(task, seed) for task in tasks for seed in task["seeds"]]
    if args.max_images:
        jobs = jobs[:args.max_images]
    variants = args.variants.split(",")
    for variant in variants:
        anchor_steps(variant)
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
                callback = SparseAnchor(
                    model, teacher_layers, teacher_final,
                    student_layers, student_final, variant)
                teacher_calls = len(callback.anchors)
                record = {"id": task["id"], "category": task["category"],
                          "prompt": task["prompt"], "seed": seed,
                          "variant": variant,
                          "teacher_velocity_calls": teacher_calls,
                          "head_block_calls_total": teacher_calls * 10 +
                                                    (49 - teacher_calls) * 5,
                          "flow_velocity_calls": args.steps - 1,
                          "gpu_physical": args.gpu_physical, "guidance": args.guidance,
                          "steps": args.steps, "checkpoint_steps": checkpoint["steps"]}
                try:
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        latents = sample_fn(z.clone(), callback, **kwargs)[-1]
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
