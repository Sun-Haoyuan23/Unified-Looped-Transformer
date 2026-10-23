"""Pilot image-level preference tuning of a looped Show-o2 prediction head.

The pretrained hybrid backbone and the recurrent visual core are frozen. Only
the 8.5M-parameter prediction head is updated. A late flow state is converted
to a one-step clean-latent estimate and decoded into an image for PickScore.
Teacher velocity MSE anchors the update. This is an exploratory training
objective; evaluation must use independent metrics and held-out prompts.
"""

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
from transformers import AutoModel, AutoProcessor

from elt_model import install_svd_loop_head, load_head_checkpoint
from final_head_tuning import install_tunable_final, save_final
from train_elt_onpolicy import guide, prepare_inputs


def pickscore_pixels(image, mean, std):
    image = ((image.float() + 1.0) * 0.5).clamp(0, 1)
    image = F.interpolate(image, size=(224, 224), mode="bicubic",
                          align_corners=False, antialias=True).clamp(0, 1)
    return (image - mean) / std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpu-physical", type=int, required=True)
    parser.add_argument("--loop-checkpoint", required=True)
    parser.add_argument("--rank", type=int, default=1280)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--reward-weight", type=float, default=0.001)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--indices", default="36,40,44,48")
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one healthy GPU by UUID")
    indices = [int(part) for part in args.indices.split(",")]
    if any(index < 0 or index >= 49 for index in indices):
        raise ValueError(indices)
    torch.set_float32_matmul_precision("high")
    source = args.work / "Show-o" / "show-o2"
    sys.path.insert(0, str(source))
    from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
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
    install_svd_loop_head(model, rank=args.rank, trainable=False)
    load_head_checkpoint(model, args.work / "stage2b" / args.loop_checkpoint)
    trainable = install_tunable_final(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in trainable:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    vae = WanVAE(vae_pth="/home/shy_23/models/show-o2-deps/Wan2.1_VAE.pth",
                 dtype=dtype, device=device)
    processor = AutoProcessor.from_pretrained(args.work / "models" / "pickscore-processor")
    reward_model = AutoModel.from_pretrained(
        args.work / "models" / "pickscore-v1", use_safetensors=True).to(device).eval()
    reward_model.requires_grad_(False)
    mean = torch.tensor(processor.image_processor.image_mean,
                        device=device).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_processor.image_std,
                       device=device).view(1, 3, 1, 1)
    scale = reward_model.logit_scale.exp().detach()

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
        text_input = processor(text=[job["prompt"]], padding=True, truncation=True,
                               max_length=77, return_tensors="pt").to(device)
        with torch.no_grad():
            text_feature = F.normalize(
                reward_model.get_text_features(**text_input).float(), dim=-1)
        records.append((payload, inputs, text_feature))
    del processor
    rng = random.Random(args.seed)
    root = args.work / "stage2b"
    metrics_path = root / f"train_{args.run_tag}.jsonl"
    print(json.dumps({"train_prompts": len(records), "indices": indices,
                      "trainable_parameters": sum(p.numel() for p in trainable),
                      "loop_checkpoint": args.loop_checkpoint}), flush=True)
    durations = []
    torch.cuda.reset_peak_memory_stats(device)
    with metrics_path.open("w", encoding="utf-8") as stream:
        for step in range(args.steps):
            payload, inputs, text_feature = records[rng.randrange(len(records))]
            index = rng.choice(indices)
            zt = payload["zt"][index].to(device)
            t_value = float(payload["t"][index])
            target = payload["teacher_velocity"][index].to(device).float()
            tokens, positions, mask = inputs
            two_zt = torch.cat((zt, zt), dim=0)
            t = torch.full((2,), t_value, device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=dtype):
                _logits, raw_velocity = model(
                    text_tokens=tokens, image_latents=two_zt, t=t,
                    attention_mask=mask, modality_positions=positions,
                    output_hidden_states=True, max_seq_len=mask.shape[-1],
                    device=device)
                velocity = guide(raw_velocity.float(), float(payload["guidance"]))
                clean_estimate = zt.float() + (1.0 - t_value) * velocity
                pixels = vae.batch_decode(clean_estimate.to(dtype).unsqueeze(2)).squeeze(2)
            pixel_values = pickscore_pixels(pixels, mean, std)
            image_feature = F.normalize(
                reward_model.get_image_features(pixel_values=pixel_values).float(), dim=-1)
            reward = scale * (text_feature * image_feature).sum(dim=-1).mean()
            anchor = F.mse_loss(velocity, target)
            loss = args.anchor_weight * anchor - args.reward_weight * reward
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step+1}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - start)
            if step == 0 or (step+1) % args.log_every == 0:
                row = {"step": step+1, "loss": float(loss.detach()),
                       "anchor_mse": float(anchor.detach()),
                       "pickscore_one_step": float(reward.detach()),
                       "grad_norm": float(grad_norm),
                       "recent_median_s": statistics.median(durations[-args.log_every:]),
                       "peak_allocated_gb": torch.cuda.max_memory_allocated(device)/1e9}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(json.dumps(row), flush=True)
            if args.save_every and (step+1) % args.save_every == 0:
                path = root / f"{args.run_tag}_step{step+1}.pt"
                save_final(model, path, step+1, args.loop_checkpoint)
                print(json.dumps({"checkpoint": str(path), "step": step+1}), flush=True)
    path = root / f"{args.run_tag}.pt"
    save_final(model, path, args.steps, args.loop_checkpoint)
    print(json.dumps({"checkpoint": str(path),
                      "median_step_s": statistics.median(durations)}), flush=True)


if __name__ == "__main__":
    main()
