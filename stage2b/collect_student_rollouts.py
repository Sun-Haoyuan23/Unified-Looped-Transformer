"""Query the original teacher on states visited by the looped student sampler."""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.nn import functional as F

from collect_trajectories import Capture
from elt_model import install_elt_head, install_svd_loop_head, load_head_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--gpu-physical", type=int, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--checkpoint", default="elt_onpolicy_5x2_bf16.pt")
    parser.add_argument("--loops", type=int, choices=(1, 2), default=2)
    parser.add_argument("--prefix-layers", type=int, default=0)
    parser.add_argument("--unique-layers", type=int, default=5)
    parser.add_argument("--architecture", choices=("shared", "svd_loop"),
                        default="shared")
    parser.add_argument("--lora-rank", type=int, default=128)
    parser.add_argument("--output-tag", default="")
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
        add_showo_tokens=True, return_showo_token_ids=True, llm_name="qwen2_5")
    model_path = str(work / "models" / "show-o2-1.5B-local")
    student = Showo2Qwen2_5.from_pretrained(
        model_path, use_safetensors=False).to(device=device, dtype=dtype).eval()
    if args.architecture == "svd_loop":
        install_svd_loop_head(student, rank=args.lora_rank, trainable=False)
    else:
        install_elt_head(student, unique_layers=args.unique_layers, loops=2,
                         prefix_layers=args.prefix_layers, precision=torch.float32,
                         trainable=False)
    checkpoint = load_head_checkpoint(student, work / "stage2b" / args.checkpoint)
    if args.loops == 1:
        student.diffusion_head_a = torch.nn.ModuleList(
            list(student.diffusion_head_a)[:args.prefix_layers + args.unique_layers])
    teacher = Showo2Qwen2_5.from_pretrained(
        model_path, use_safetensors=False).to(device=device, dtype=dtype).eval()
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
        sampling_method=config.transport.sampling_method, num_steps=50,
        atol=config.transport.atol, rtol=config.transport.rtol,
        reverse=config.transport.reverse,
        time_shifting_factor=config.transport.time_shifting_factor)
    jobs = json.loads(args.manifest.read_text(encoding="utf-8"))[args.split]
    jobs = [job for i, job in enumerate(jobs) if i % args.num_shards == args.shard]
    if args.max_prompts:
        jobs = jobs[:args.max_prompts]
    root = work / "stage2b"
    directory_name = args.output_tag or f"student_L{args.loops}_{args.split}"
    save_dir = root / "trajectories" / directory_name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stream_path = log_dir / f"collect_{directory_name}_gpu{args.gpu_physical}.jsonl"
    print(json.dumps({"gpu": args.gpu_physical, "jobs": len(jobs),
                      "student_loops": args.loops,
                      "checkpoint_steps": checkpoint["steps"]}), flush=True)
    with stream_path.open("w", encoding="utf-8") as stream:
        for job in jobs:
            row = {"id": job["id"], "gpu_physical": args.gpu_physical,
                   "student_loops": args.loops}
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
                              max_seq_len=max_seq_len, guidance_scale=5.0)
                generator = torch.Generator(device=device).manual_seed(job["seed"])
                noise = torch.randn((1, channels, latent_h * patch_size,
                                     latent_w * patch_size), generator=generator,
                                    device=device, dtype=dtype)
                z = torch.cat((noise, noise), dim=0)
                capture = Capture(student)
                with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                    path = sample_fn(z, capture, **kwargs)
                if len(capture.times) != 49:
                    raise RuntimeError(f"Expected 49 calls; got {len(capture.times)}")
                teacher_velocity = []
                with torch.no_grad():
                    for zt, t in zip(capture.latents, capture.times):
                        zz = torch.cat((zt, zt), dim=0)
                        tt = torch.full((2,), float(t), device=device, dtype=torch.float32)
                        v = teacher.t2i_generate(zz, tt, **kwargs)
                        teacher_velocity.append(v[:1].detach().to(dtype).clone())
                student_velocities = torch.stack(capture.velocities).cpu()
                teacher_velocities = torch.stack(teacher_velocity).cpu()
                payload = {"id": job["id"], "prompt": job["prompt"],
                           "category": job["category"], "challenge": job["challenge"],
                           "seed": job["seed"], "guidance": 5.0, "steps": 50,
                           "rollout_loops": args.loops,
                           "zt": torch.stack(capture.latents).cpu(),
                           "t": torch.stack(capture.times).cpu(),
                           "teacher_velocity": teacher_velocities,
                           "student_velocity": student_velocities,
                           "final_latent": path[-1, :1].to(dtype).cpu()}
                file = save_dir / f"{job['id']}.pt"
                torch.save(payload, file)
                row.update({"status": "ok", "path": str(file),
                            "mean_student_teacher_mse": round(float(F.mse_loss(
                                student_velocities.float(),
                                teacher_velocities.float())), 6),
                            "time_s": round(time.perf_counter() - started, 4)})
            except Exception as error:
                row.update({"status": "error", "error": str(error),
                            "traceback": traceback.format_exc(limit=5)})
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps({k: row.get(k) for k in
                              ("id", "status", "mean_student_teacher_mse",
                               "time_s", "error")}), flush=True)


if __name__ == "__main__":
    main()
