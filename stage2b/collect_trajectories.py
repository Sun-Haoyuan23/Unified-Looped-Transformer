"""Collect on-policy Show-o2 latent/velocity pairs at every actual ODE call."""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch
from omegaconf import OmegaConf


class Capture:
    def __init__(self, model):
        self.model = model
        self.latents = []
        self.times = []
        self.velocities = []

    def __call__(self, image_latents, t, **kwargs):
        result = self.model.t2i_generate(image_latents, t, **kwargs)
        # CFG's two branches have the same latent and guided output.
        self.latents.append(image_latents[:1].detach().to(torch.bfloat16).clone())
        self.times.append(t[0].detach().float().clone())
        self.velocities.append(result[:1].detach().to(torch.bfloat16).clone())
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--gpu-physical", type=int, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose one healthy physical GPU by UUID")
    work = args.work
    source = work / "Show-o" / "show-o2"
    sys.path.insert(0, str(source))
    from models import Showo2Qwen2_5, omni_attn_mask_naive
    from models.misc import get_text_tokenizer, prepare_gen_input
    from transport import Sampler, create_transport
    from utils import get_hyper_params

    device, dtype = torch.device("cuda:0"), torch.bfloat16
    config = OmegaConf.load(source / "configs" / "showo2_1.5b_demo_432x432.yaml")
    tokenizer, ids = get_text_tokenizer(
        str(work / "models" / "qwen2.5-1.5b-tokenizer"),
        add_showo_tokens=True, return_showo_token_ids=True, llm_name="qwen2_5"
    )
    model = Showo2Qwen2_5.from_pretrained(
        str(work / "models" / "show-o2-1.5B-local"), use_safetensors=False
    ).to(device=device, dtype=dtype).eval()
    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1
    (nvisual, _nmmu, _nvideo, max_seq_len, max_text_len, channels, patch_size,
     latent_w, latent_h, pad_id, bos_id, eos_id, boi_id, eoi_id,
     _bov, _eov, img_pad_id, _vid, _guide) = get_hyper_params(config, tokenizer, ids)
    transport = create_transport(
        path_type=config.transport.path_type, prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight, train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps, snr_type=config.transport.snr_type,
        do_shift=config.transport.do_shift, seq_len=nvisual)
    sample_fn = Sampler(transport).sample_ode(
        sampling_method=config.transport.sampling_method, num_steps=args.steps,
        atol=config.transport.atol, rtol=config.transport.rtol,
        reverse=config.transport.reverse,
        time_shifting_factor=config.transport.time_shifting_factor)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = [job for i, job in enumerate(manifest[args.split])
            if i % args.num_shards == args.shard]
    if args.max_prompts:
        jobs = jobs[:args.max_prompts]
    root = work / "stage2b"
    save_dir = root / "trajectories" / args.split
    log_dir = root / "logs"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics = log_dir / f"collect_{args.split}_gpu{args.gpu_physical}.jsonl"
    print(json.dumps({"gpu": args.gpu_physical, "jobs": len(jobs),
                      "split": args.split}), flush=True)
    with metrics.open("w", encoding="utf-8") as stream:
        for job in jobs:
            file = save_dir / f"{job['id']}.pt"
            if args.skip_existing and file.exists():
                continue
            row = {"id": job["id"], "split": args.split,
                   "gpu_physical": args.gpu_physical}
            started = time.perf_counter()
            try:
                cond, uncond, cpos, upos = prepare_gen_input(
                    [job["prompt"]], tokenizer, nvisual, bos_id, eos_id,
                    boi_id, eoi_id, pad_id, img_pad_id, max_text_len, device)
                tokens = torch.cat((cond, uncond), dim=0)
                positions = torch.cat((cpos, upos), dim=0)
                mask = omni_attn_mask_naive(2, max_seq_len, positions, device).to(dtype)
                kwargs = dict(text_tokens=tokens, attention_mask=mask,
                              modality_positions=positions, output_hidden_states=True,
                              max_seq_len=max_seq_len, guidance_scale=args.guidance)
                generator = torch.Generator(device=device).manual_seed(job["seed"])
                noise = torch.randn((1, channels, latent_h * patch_size,
                                     latent_w * patch_size), generator=generator,
                                    device=device, dtype=dtype)
                z = torch.cat((noise, noise), dim=0)
                capture = Capture(model)
                with torch.no_grad():
                    path = sample_fn(z, capture, **kwargs)
                if len(capture.times) != args.steps - 1:
                    raise RuntimeError(f"Expected {args.steps-1} calls; got {len(capture.times)}")
                payload = {"id": job["id"], "prompt": job["prompt"],
                           "category": job["category"], "challenge": job["challenge"],
                           "seed": job["seed"], "guidance": args.guidance,
                           "steps": args.steps,
                           "zt": torch.stack(capture.latents).cpu(),
                           "t": torch.stack(capture.times).cpu(),
                           "teacher_velocity": torch.stack(capture.velocities).cpu(),
                           "final_latent": path[-1, :1].to(torch.bfloat16).cpu()}
                torch.save(payload, file)
                row.update({"status": "ok", "path": str(file),
                            "calls": len(capture.times), "bytes": file.stat().st_size,
                            "time_s": round(time.perf_counter()-started, 4)})
            except Exception as error:
                row.update({"status": "error", "error": str(error),
                            "traceback": traceback.format_exc(limit=5)})
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps({k: row.get(k) for k in ("id", "status", "time_s", "error")}),
                  flush=True)


if __name__ == "__main__":
    main()
