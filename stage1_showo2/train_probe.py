"""200-step rank-16 flow-head LoRA throughput probe on generated images.

The 14 baseline outputs are re-encoded with the official Wan VAE. The probe
uses Show-o2's real flow-matching path and measures training time/memory only;
it is not a quality, preference, or generalization experiment.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = 1.0
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.normal_(self.a, std=0.01)

    def forward(self, x):
        residual = F.linear(F.linear(x.float(), self.a), self.b).to(x.dtype)
        return self.base(x) + self.scale * residual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--gpu-physical", type=int, required=True)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU via CUDA_VISIBLE_DEVICES")
    sys.path.insert(0, str(args.work / "Show-o" / "show-o2"))
    from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
    from models.misc import get_text_tokenizer, prepare_gen_input
    from transport import create_transport
    from utils import get_hyper_params

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    config = OmegaConf.load(args.work / "Show-o" / "show-o2" / "configs" /
                            "showo2_1.5b_demo_432x432.yaml")
    tokenizer, token_ids = get_text_tokenizer(
        str(args.work / "models" / "qwen2.5-1.5b-tokenizer"),
        add_showo_tokens=True, return_showo_token_ids=True, llm_name="qwen2_5"
    )
    model = Showo2Qwen2_5.from_pretrained(
        str(args.work / "models" / "show-o2-1.5B-local"), use_safetensors=False
    ).to(device=device, dtype=dtype).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    lora_parameters = []
    for block in model.diffusion_head_a:
        for name in ("q_proj", "v_proj"):
            original = getattr(block.self_attn, name)
            wrapped = LoRALinear(original, rank=16)
            setattr(block.self_attn, name, wrapped)
            lora_parameters.extend((wrapped.a, wrapped.b))
    trainable_count = sum(p.numel() for p in lora_parameters)
    optimizer = torch.optim.AdamW(lora_parameters, lr=1e-4)
    vae = WanVAE(
        vae_pth="/home/shy_23/models/show-o2-deps/Wan2.1_VAE.pth",
        dtype=dtype, device=device
    )

    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1
    (nvisual, _nmmu, _nvideo, max_seq_len, max_text_len, _channels, _patch,
     _w, _h, pad_id, bos_id, eos_id, boi_id, eoi_id,
     _bov, _eov, img_pad_id, _vid, _scale) = get_hyper_params(config, tokenizer, token_ids)
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

    records = []
    for task in json.loads((args.work / "prompts.json").read_text(encoding="utf-8")):
        image_path = args.work / "images" / f"{task['id']}_seed{task['seeds'][0]}.png"
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
        pixels = torch.from_numpy(image).permute(2, 0, 1)[None].to(device=device, dtype=dtype)
        pixels = pixels / 127.5 - 1.0
        with torch.no_grad():
            z1 = vae.sample(pixels.unsqueeze(2), deterministic=True).squeeze(2).detach()
        tokens, _null, positions, _null_pos = prepare_gen_input(
            [task["prompt"]], tokenizer, nvisual, bos_id, eos_id, boi_id,
            eoi_id, pad_id, img_pad_id, max_text_len, device
        )
        mask = omni_attn_mask_naive(1, max_seq_len, positions, device).to(dtype)
        image_mask = torch.zeros((1, max_seq_len), device=device, dtype=torch.float32)
        offset, length = positions[0, 0].tolist()
        image_mask[:, offset + 1:offset + length] = 1.0
        records.append((z1, tokens, positions, mask, image_mask))
    del vae
    torch.cuda.empty_cache()

    torch.manual_seed(20261003)
    times, losses = [], []
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.steps):
        z1, tokens, positions, attention_mask, image_mask = records[step % len(records)]
        t, noise, clean = transport.sample(z1)
        t, zt, target_velocity = transport.path_sampler.plan(t, noise, clean)
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        _logits, loss = model(
            text_tokens=tokens,
            image_latents=zt,
            t=t.to(dtype),
            attention_mask=attention_mask,
            image_masks=image_mask,
            image_labels=target_velocity,
            modality_positions=positions,
            output_hidden_states=True,
            max_seq_len=max_seq_len,
            device=device,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        losses.append(float(loss.detach().item()))
        if (step + 1) % 20 == 0:
            print(json.dumps({"step": step + 1, "loss": round(losses[-1], 4),
                              "recent_median_step_s": round(statistics.median(times[-20:]), 3)}),
                  flush=True)

    stable_times = times[min(20, len(times) // 4):]
    result = {
        "purpose": "training throughput probe only; baseline-generated images are re-encoded",
        "physical_gpu": args.gpu_physical,
        "steps": args.steps,
        "microbatch": 1,
        "rank": 16,
        "trainable_parameters": trainable_count,
        "training_images": len(records),
        "first_loss": round(losses[0], 6),
        "last_loss": round(losses[-1], 6),
        "median_step_s_after_warmup": round(statistics.median(stable_times), 4),
        "mean_step_s_after_warmup": round(statistics.mean(stable_times), 4),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved(device) / 1e9, 3),
    }
    output = args.work / "logs" / f"train_probe_gpu{args.gpu_physical}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
