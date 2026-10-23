"""Run the released Show-o2 1.5B Euler-flow T2I path and record stage-1 costs.

This script is intentionally limited to inference. It uses the official
Show-o2 model, tokenizer preparation, attention mask, transport and Wan VAE.
"""

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image


class ComponentTimer:
    def __init__(self, model):
        self.events = defaultdict(list)
        self.starts = defaultdict(list)
        self.handles = []
        components = {
            "language_backbone": [model.showo],
            "semantic_transformer": [model.und_trans],
            "visual_embedders": [model.image_embedder_und, model.image_embedder_gen],
            "fusion": [model.fusion_proj],
            "flow_head": list(model.diffusion_head_a) + [model.diffusion_head_b],
        }
        if hasattr(model, "diff_proj"):
            components["flow_head"].append(model.diff_proj)
        for category, modules in components.items():
            for module in modules:
                self.handles.append(module.register_forward_pre_hook(self._pre(category)))
                self.handles.append(module.register_forward_hook(self._post(category)))

    def _pre(self, category):
        def hook(module, inputs):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.starts[id(module)].append(event)
        return hook

    def _post(self, category):
        def hook(module, inputs, output):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            start = self.starts[id(module)].pop()
            self.events[category].append((start, end))
        return hook

    def result(self):
        torch.cuda.synchronize()
        return {
            category: {"ms": round(sum(a.elapsed_time(b) for a, b in pairs), 3), "calls": len(pairs)}
            for category, pairs in self.events.items()
        }

    def close(self):
        for handle in self.handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--gpu-physical", type=int, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--profile-first", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tasks-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--metrics-tag", default="")
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to one physical GPU for each worker")

    source = args.work / "Show-o" / "show-o2"
    sys.path.insert(0, str(source))
    from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
    from models.misc import get_text_tokenizer, prepare_gen_input
    from transport import Sampler, create_transport
    from utils import denorm, get_hyper_params

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    config = OmegaConf.load(source / "configs" / "showo2_1.5b_demo_432x432.yaml")
    tokenizer_path = args.work / "models" / "qwen2.5-1.5b-tokenizer"
    checkpoint_path = args.work / "models" / "show-o2-1.5B-local"
    vae_path = Path("/home/shy_23/models/show-o2-deps/Wan2.1_VAE.pth")

    start_load = time.perf_counter()
    tokenizer, token_ids = get_text_tokenizer(
        str(tokenizer_path), add_showo_tokens=True,
        return_showo_token_ids=True, llm_name="qwen2_5"
    )
    if len(tokenizer) != 151669:
        raise RuntimeError(f"Unexpected tokenizer length: {len(tokenizer)}")
    config.model.showo.llm_vocab_size = len(tokenizer)
    model = Showo2Qwen2_5.from_pretrained(str(checkpoint_path), use_safetensors=False)
    meta_parameters = [name for name, value in model.named_parameters() if value.is_meta]
    if meta_parameters:
        raise RuntimeError(f"Checkpoint left meta parameters unmaterialized: {meta_parameters[:8]}")
    model = model.to(device=device, dtype=dtype).eval()
    vae = WanVAE(vae_pth=str(vae_path), dtype=dtype, device=device)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - start_load

    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1

    (nvisual, _nmmu, _nvideo, max_seq_len, max_text_len, channels, patch_size,
     latent_w, latent_h, pad_id, bos_id, eos_id, boi_id, eoi_id,
     _bov_id, _eov_id, img_pad_id, _vid_pad_id, _config_guidance) = get_hyper_params(
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
    if args.max_images > 0:
        jobs = jobs[:args.max_images]

    output_dir = args.output_dir or args.work / ("smoke" if args.smoke else "images")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.work / "logs" / (
        f"{args.metrics_tag}_gpu{args.gpu_physical}.jsonl" if args.metrics_tag else
        (f"smoke_gpu{args.gpu_physical}.jsonl" if args.smoke else
         f"metrics_gpu{args.gpu_physical}.jsonl"))
    run_info = {
        "physical_gpu": args.gpu_physical,
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "model_load_s": round(load_s, 3),
        "model_checkpoint": str(checkpoint_path),
        "steps": args.steps,
        "guidance": args.guidance,
        "dtype": "bfloat16",
        "scheduler": "official shifted Euler flow",
        "resolution": 432,
        "safety_checker": "not applied; raw model output",
        "task_count": len(jobs),
    }
    (args.work / "logs" / f"run_info_gpu{args.gpu_physical}.json").write_text(
        json.dumps(run_info, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_info), flush=True)

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for index, (task, seed) in enumerate(jobs):
            timer = ComponentTimer(model) if args.profile_first and index == 0 else None
            record = {"id": task["id"], "category": task["category"],
                      "prompt": task["prompt"], "seed": seed,
                      "gpu_physical": args.gpu_physical, "steps": args.steps,
                      "guidance": args.guidance, "profiled": timer is not None}
            try:
                torch.cuda.reset_peak_memory_stats(device)
                start_prepare = time.perf_counter()
                (cond, uncond, cond_pos, uncond_pos) = prepare_gen_input(
                    [task["prompt"]], tokenizer, nvisual, bos_id, eos_id,
                    boi_id, eoi_id, pad_id, img_pad_id, max_text_len, device
                )
                noise_rng = torch.Generator(device=device).manual_seed(seed)
                z = torch.randn(
                    (1, channels, latent_h * patch_size, latent_w * patch_size),
                    generator=noise_rng, device=device, dtype=dtype
                )
                if args.guidance > 0:
                    z = torch.cat((z, z), dim=0)
                    tokens = torch.cat((cond, uncond), dim=0)
                    positions = torch.cat((cond_pos, uncond_pos), dim=0)
                else:
                    tokens, positions = cond, cond_pos
                mask = omni_attn_mask_naive(
                    tokens.size(0), max_seq_len, positions, device
                ).to(dtype)
                torch.cuda.synchronize()
                prepare_s = time.perf_counter() - start_prepare

                model_kwargs = dict(
                    text_tokens=tokens, attention_mask=mask,
                    modality_positions=positions, output_hidden_states=True,
                    max_seq_len=max_seq_len, guidance_scale=args.guidance
                )
                start_denoise = time.perf_counter()
                with torch.no_grad():
                    latents = sample_fn(z, model.t2i_generate, **model_kwargs)[-1]
                torch.cuda.synchronize()
                denoise_s = time.perf_counter() - start_denoise
                if args.guidance > 0:
                    latents = torch.chunk(latents, 2, dim=0)[0]

                start_decode = time.perf_counter()
                with torch.no_grad():
                    image_tensor = vae.batch_decode(latents.unsqueeze(2)).squeeze(2)
                image_array = denorm(image_tensor)[0]
                image_path = output_dir / f"{task['id']}_seed{seed}.png"
                Image.fromarray(image_array).save(image_path)
                torch.cuda.synchronize()
                decode_s = time.perf_counter() - start_decode

                record.update({
                    "status": "ok", "image_path": str(image_path),
                    "prepare_s": round(prepare_s, 4),
                    "denoise_s": round(denoise_s, 4),
                    "decode_and_save_s": round(decode_s, 4),
                    "total_s": round(prepare_s + denoise_s + decode_s, 4),
                    "peak_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 3),
                    "peak_reserved_gb": round(torch.cuda.max_memory_reserved(device) / 1e9, 3),
                })
                if timer:
                    record["component_gpu_time"] = timer.result()
            except Exception as error:
                record.update({"status": "error", "error": str(error),
                               "traceback": traceback.format_exc(limit=5)})
                print(json.dumps(record), flush=True)
            finally:
                if timer:
                    timer.close()
                metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                metrics_file.flush()
            if record["status"] == "ok":
                print(json.dumps({k: record[k] for k in
                                  ["id", "seed", "gpu_physical", "total_s", "peak_allocated_gb"]}),
                      flush=True)


if __name__ == "__main__":
    main()
