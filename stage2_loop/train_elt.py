"""ELT-inspired, cost-matched Show-o2 flow-head pilot with intra-loop distillation.

Original head: 10 unique blocks. Student: 4 unique prefix blocks followed by
2 shared blocks repeated 1, 2, or 3 times. Maximum depth is 10 block calls;
intermediate exits use 6 or 8 calls. Only rank-16 q/v/output LoRA is trained.
Data are re-encoded Show-o2 baseline images, so this is a functional pilot.
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(rank, base.in_features,
                                          device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank,
                                          device=base.weight.device, dtype=torch.float32))
        nn.init.normal_(self.a, std=0.01)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x.float(), self.a), self.b).to(x.dtype)


def install_tied_tail(model, rank=16):
    original = list(model.diffusion_head_a)
    if len(original) != 10:
        raise RuntimeError(f"Expected 10 head blocks, found {len(original)}")
    prefix, shared = original[:4], original[4:6]
    model.diffusion_head_a = nn.ModuleList(prefix + shared * 3)
    for param in model.parameters():
        param.requires_grad_(False)
    for block in shared:
        for name in ("q_proj", "v_proj"):
            setattr(block.self_attn, name,
                    LoRALinear(getattr(block.self_attn, name), rank=rank))
    model.diffusion_head_b.linear = LoRALinear(model.diffusion_head_b.linear, rank=rank)
    adapters = [p for p in model.parameters() if p.requires_grad]
    return adapters


class LoopCapture:
    def __init__(self, model):
        self.hidden = []
        self.handle = model.diffusion_head_a[5].register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.hidden.append(output[0])

    def close(self):
        self.handle.remove()


def token_velocity(model, hidden, time_embed, positions, channels, height, width):
    tokens = model.diffusion_head_b(hidden, time_embed, positions)
    offset, length = positions[0, 0].tolist()
    start = offset + int(model.config.add_time_embeds)
    visual = tokens[:, start:offset + length]
    flat = model.unpatchify(visual, height // model.config.patch_size,
                            width // model.config.patch_size)
    return flat.transpose(1, 2).reshape(hidden.shape[0], channels, height, width)


def forward_all(model, capture, record, zt, t):
    z1, tokens, positions, mask = record
    capture.hidden.clear()
    _logits, final_velocity = model(
        text_tokens=tokens, image_latents=zt, t=t.to(torch.bfloat16),
        attention_mask=mask, modality_positions=positions,
        output_hidden_states=True, max_seq_len=mask.shape[-1], device=zt.device,
    )
    if len(capture.hidden) != 3:
        raise RuntimeError(f"Expected 3 loop exits, got {len(capture.hidden)}")
    time_embed = model.time_embed(t.to(torch.bfloat16), torch.bfloat16)
    channels, height, width = zt.shape[1:]
    first = token_velocity(model, capture.hidden[0], time_embed, positions,
                           channels, height, width)
    second = token_velocity(model, capture.hidden[1], time_embed, positions,
                            channels, height, width)
    return first, second, final_velocity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--gpu-physical", type=int, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose one GPU by UUID")
    source = args.work / "Show-o" / "show-o2"
    sys.path.insert(0, str(source))
    from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
    from models.misc import get_text_tokenizer, prepare_gen_input
    from transport import create_transport
    from utils import get_hyper_params

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
    adapters = install_tied_tail(model, rank=args.rank)
    capture = LoopCapture(model)
    optimizer = torch.optim.AdamW(adapters, lr=args.lr, weight_decay=0.01)
    vae = WanVAE(vae_pth="/home/shy_23/models/show-o2-deps/Wan2.1_VAE.pth",
                 dtype=dtype, device=device)

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
    records_train, records_val = [], []
    tasks = json.loads((args.work / "prompts.json").read_text(encoding="utf-8"))
    for task in tasks:
        tokens, _null, positions, _null_pos = prepare_gen_input(
            [task["prompt"]], tokenizer, nvisual, bos_id, eos_id,
            boi_id, eoi_id, pad_id, img_pad_id, max_text_len, device
        )
        mask = omni_attn_mask_naive(1, max_seq_len, positions, device).to(dtype)
        for i, seed in enumerate(task["seeds"]):
            path = args.work / "images" / f"{task['id']}_seed{seed}.png"
            image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
            pixels = torch.from_numpy(image).permute(2, 0, 1)[None].to(device, dtype)
            pixels = pixels / 127.5 - 1.0
            with torch.no_grad():
                z1 = vae.sample(pixels.unsqueeze(2), deterministic=True).squeeze(2).detach()
            record = (z1, tokens, positions, mask)
            (records_val if i == 3 else records_train).append(record)
    del vae
    torch.cuda.empty_cache()
    print(json.dumps({"train_images": len(records_train), "val_images": len(records_val),
                      "trainable_parameters": sum(p.numel() for p in adapters)}), flush=True)

    torch.manual_seed(20261004)
    validation = []
    for record in records_val:
        t, noise, clean = transport.sample(record[0])
        t, zt, target = transport.path_sampler.plan(t, noise, clean)
        validation.append((record, zt, t, target))

    def validate():
        output = [[], [], []]
        with torch.no_grad():
            for record, zt, t, target in validation:
                velocities = forward_all(model, capture, record, zt, t)
                for index, velocity in enumerate(velocities):
                    output[index].append(float(F.mse_loss(velocity.float(), target.float())))
        return [round(statistics.mean(group), 6) for group in output]

    before = validate()
    rng = random.Random(20261003)
    times, recent = [], []
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.steps):
        record = records_train[rng.randrange(len(records_train))]
        t, noise, clean = transport.sample(record[0])
        t, zt, target = transport.path_sampler.plan(t, noise, clean)
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        first, second, final = forward_all(model, capture, record, zt, t)
        intermediate = first if rng.random() < 0.5 else second
        target_float = target.float()
        loss_final = F.mse_loss(final.float(), target_float)
        loss_intermediate = F.mse_loss(intermediate.float(), target_float)
        loss_distill = F.mse_loss(intermediate.float(), final.detach().float())
        loss = loss_final + 0.5 * loss_intermediate + 0.5 * loss_distill
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapters, 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
        recent.append(float(loss.detach()))
        if (step + 1) % 50 == 0:
            print(json.dumps({"step": step + 1, "loss": round(recent[-1], 5),
                              "median_recent_step_s": round(statistics.median(times[-50:]), 4)}),
                  flush=True)
    after = validate()
    result_dir = args.work / "stage2_loop"
    checkpoint = result_dir / "elt_tied_tail_rank16.pt"
    state = {name: param.detach().cpu() for name, param in model.named_parameters()
             if param.requires_grad}
    torch.save({"adapters": state, "steps": args.steps, "rank": args.rank,
                "prefix_layers": 4, "shared_layers": 2, "max_loops": 3,
                "source": "Show-o2 baseline-generated images re-encoded by Wan VAE"}, checkpoint)
    summary = {
        "purpose": "functional ELT-inspired pilot, not human preference evidence",
        "physical_gpu": args.gpu_physical, "steps": args.steps,
        "train_images": len(records_train), "validation_images": len(records_val),
        "trainable_parameters": sum(p.numel() for p in adapters),
        "head_block_calls": {"baseline": 10, "L1": 6, "L2": 8, "L3": 10},
        "validation_velocity_mse_before_L1_L2_L3": before,
        "validation_velocity_mse_after_L1_L2_L3": after,
        "median_step_s_after_warmup": round(statistics.median(times[min(20, len(times)//4):]), 4),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 3),
        "checkpoint": str(checkpoint),
    }
    (result_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                    encoding="utf-8")
    print(json.dumps(summary), flush=True)
    capture.close()


if __name__ == "__main__":
    main()
