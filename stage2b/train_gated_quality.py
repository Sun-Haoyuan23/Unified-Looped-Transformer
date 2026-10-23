"""Quality-aware pilot for a zero-gated third visit of a shared visual core.

Only a 2048-dimensional gate is trained. A differentiable PickScore signal
acts on predicted clean images at late flow times. Either real-image velocity
or on-policy released-model velocity anchors the update. Held-out generation,
not training reward, determines whether this direction is useful.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, AutoProcessor

from gated_revisit import insert_gated_revisit, save_gate
from train_coco_recurrence import load_record
from train_elt_onpolicy import prepare_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--rollout-steps", type=int, default=0,
                        help="Backpropagate through this many final Euler steps")
    parser.add_argument("--checkpoint-rollout", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--reward-weight", type=float, default=0.001)
    parser.add_argument("--flow-weight", type=float, default=1.0)
    parser.add_argument("--max-gain", type=float, default=0.1)
    parser.add_argument("--revisit-position", type=int, choices=(5, 8), default=8,
                        help="5 repeats blocks 2-4 immediately; 8 is the legacy late revisit")
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args()
    if args.rollout_steps and not args.trajectory_dir:
        parser.error("--rollout-steps requires --trajectory-dir")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose one healthy GPU by UUID")
    source = args.work / "Show-o" / "show-o2"
    sys.path.insert(0, str(source))
    from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
    from models.misc import get_text_tokenizer
    from utils import get_hyper_params

    device = torch.device("cuda:0")
    config = OmegaConf.load(source / "configs" / "showo2_1.5b_demo_432x432.yaml")
    tokenizer, ids = get_text_tokenizer(
        str(args.work / "models" / "qwen2.5-1.5b-tokenizer"),
        add_showo_tokens=True, return_showo_token_ids=True, llm_name="qwen2_5")
    model = Showo2Qwen2_5.from_pretrained(
        str(args.work / "models" / "show-o2-1.5B-local"),
        use_safetensors=False).to(device=device, dtype=torch.bfloat16).eval()
    gate = insert_gated_revisit(model, max_gain=args.max_gain,
                                position=args.revisit_position)
    vae = WanVAE(vae_pth="/home/shy_23/models/show-o2-deps/Wan2.1_VAE.pth",
                 dtype=torch.bfloat16, device=device)
    processor = AutoProcessor.from_pretrained(
        args.work / "models" / "pickscore-processor")
    scorer = AutoModel.from_pretrained(
        args.work / "models" / "pickscore-v1",
        use_safetensors=True).eval().requires_grad_(False).to(device)
    mean = torch.tensor(processor.image_processor.image_mean,
                        device=device).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_processor.image_std,
                       device=device).view(1, 3, 1, 1)
    image_size = processor.image_processor.crop_size["height"]
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
    if args.trajectory_dir:
        records = sorted(args.trajectory_dir.glob("*.pt"))
    else:
        manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
        records = [row for row in manifest["records"] if row["split"] == "train"]
    if not records:
        raise RuntimeError("No training records")
    rng = random.Random(20261003)
    optimizer = torch.optim.AdamW([gate.gate], lr=args.lr, weight_decay=0)
    output = args.work / "stage2b"
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        if args.trajectory_dir:
            payload = torch.load(rng.choice(records), map_location="cpu",
                                 weights_only=False)
            row = {"prompt": payload["prompt"]}
            if args.rollout_steps:
                if args.rollout_steps > len(payload["t"]):
                    raise RuntimeError("Rollout longer than trajectory")
                index = len(payload["t"]) - args.rollout_steps
                t_value = float(payload["t"][index])
                zt = payload["zt"][index].to(device).float()
                target = payload["final_latent"].to(device).float()
            else:
                indices = [i for i, value in enumerate(payload["t"])
                           if 0.65 <= float(value) <= 0.95]
                if not indices:
                    raise RuntimeError("Trajectory lacks late flow states")
                index = rng.choice(indices)
                t_value = float(payload["t"][index])
                zt = payload["zt"][index].to(device).float()
                target = payload["teacher_velocity"][index].to(device).float()
            guidance = float(payload["guidance"])
        else:
            row = rng.choice(records)
            z = load_record(row, args.data, device)
            t_value = rng.uniform(0.65, 0.95)
            noise = torch.randn_like(z)
            zt = (1 - t_value) * noise + t_value * z
            target = (z - noise).float()
            guidance = None
        inputs = prepare_inputs(row, tokenizer, hyper,
                                omni_attn_mask_naive, device, torch.bfloat16)
        tokenized = processor(text=[row["prompt"]], padding=True,
                              truncation=True, max_length=77,
                              return_tensors="pt").to(device)
        with torch.no_grad():
            text_features = scorer.get_text_features(**tokenized).float()
            text_features = F.normalize(text_features, dim=-1)
        tokens, positions, mask = inputs
        optimizer.zero_grad(set_to_none=True)
        if args.rollout_steps:
            clean_prediction = zt
            for offset in range(args.rollout_steps):
                trajectory_index = index + offset
                time_now = float(payload["t"][trajectory_index])
                time_next = (float(payload["t"][trajectory_index + 1])
                             if trajectory_index + 1 < len(payload["t"]) else 1.0)
                t = torch.full((2,), time_now, device=device)
                def velocity_raw(latents, time_tensor):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        return model(
                            text_tokens=tokens, image_latents=latents,
                            t=time_tensor, attention_mask=mask,
                            modality_positions=positions,
                            output_hidden_states=True,
                            max_seq_len=mask.shape[-1], device=device)[1]
                paired = torch.cat((clean_prediction, clean_prediction))
                raw = (checkpoint(velocity_raw, paired, t,
                                  use_reentrant=False)
                       if args.checkpoint_rollout else velocity_raw(paired, t))
                velocity = raw[1:].float() + guidance * (
                    raw[:1].float() - raw[1:].float())
                clean_prediction = clean_prediction + (time_next - time_now) * velocity
            flow_loss = F.mse_loss(clean_prediction, target)
        else:
            t = torch.full((2,), t_value, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _logits, raw = model(
                    text_tokens=tokens, image_latents=torch.cat((zt, zt)),
                    t=t, attention_mask=mask, modality_positions=positions,
                    output_hidden_states=True, max_seq_len=mask.shape[-1],
                    device=device)
            velocity = (raw[:1].float() if guidance is None else
                        raw[1:].float() + guidance *
                        (raw[:1].float() - raw[1:].float()))
            flow_loss = F.mse_loss(velocity, target)
            clean_prediction = zt.float() + (1 - t_value) * velocity
        image = vae.batch_decode(clean_prediction.unsqueeze(2)).squeeze(2)
        image = (image + 1).mul(0.5).clamp(0, 1)
        image = F.interpolate(image, size=(image_size, image_size),
                              mode="bicubic", align_corners=False,
                              antialias=True)
        image = (image - mean) / std
        image_features = scorer.get_image_features(pixel_values=image).float()
        image_features = F.normalize(image_features, dim=-1)
        reward = scorer.logit_scale.exp() * (text_features * image_features).sum()
        gain = args.max_gain * gate.gate.tanh()
        loss = (args.flow_weight * flow_loss - args.reward_weight * reward +
                0.1 * gain.square().mean())
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([gate.gate], 1.0)
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(json.dumps({"step": step, "loss": float(loss),
                              "flow": float(flow_loss), "pickscore": float(reward),
                              "grad_norm": float(grad_norm),
                              "gain_rms": float(gain.square().mean().sqrt()),
                              "peak_gb": round(torch.cuda.max_memory_allocated()/1e9, 3),
                              "elapsed_s": round(time.perf_counter()-started, 1)}),
                  flush=True)
        if step in (25, 50, 100, args.steps):
            save_gate(gate, output / f"{args.run_tag}_step{step}.pt", step)


if __name__ == "__main__":
    main()
