"""Diagnostic: ask Show-o2's own understanding path about paired generations.

This measures whether generated images support their prompt more than a
mismatched prompt. It is not an understanding-transfer or VQA benchmark.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image


def ask(model, tokenizer, token_ids, visual, question, nvisual, device, dtype):
    system = tokenizer("system\nYou are a helpful assistant.<|im_end|>",
                       add_special_tokens=False).input_ids
    user = tokenizer("\n<|im_start|>user\n", add_special_tokens=False).input_ids
    assistant = tokenizer("\n<|im_start|>assistant\n",
                          add_special_tokens=False).input_ids
    query = tokenizer(question, add_special_tokens=False).input_ids
    a = torch.tensor([token_ids["bos_id"]] + system + user,
                     device=device)[None]
    b = torch.tensor([token_ids["boi_id"], token_ids["eoi_id"]] + query + assistant,
                     device=device)[None]
    a_embed = model.showo.model.embed_tokens(a)
    b_embed = model.showo.model.embed_tokens(b)
    time_embed = model.time_embed(torch.tensor([[1.0]], device=device),
                                 a_embed.dtype)
    if hasattr(model, "time_embed_proj"):
        time_embed = model.time_embed_proj(time_embed)
    inputs = torch.cat((a_embed, b_embed[:, :1], time_embed,
                        visual, b_embed[:, 1:]), dim=1).to(dtype)
    from models import omni_attn_mask_naive
    positions = torch.tensor([a.shape[1] + 2, nvisual], device=device)[None, None]
    mask = omni_attn_mask_naive(B=1, LEN=inputs.shape[1],
                                modalities=positions, device=device,
                                inverted=True).to(inputs.dtype)
    output = model.mmu_generate(input_embeds=inputs, attention_mask=mask,
                                top_k=1, max_new_tokens=12,
                                eos_token=tokenizer.eos_token_id)
    tokens = torch.stack(output).squeeze().reshape(1, -1)
    return tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()


def classify(answer):
    words = answer.lower().lstrip(" .,:;\n")
    if words.startswith("yes"):
        return 1
    if words.startswith("no"):
        return 0
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--tasks-file", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    source = args.work / "Show-o" / "show-o2"
    sys.path.insert(0, str(source))
    from models import Showo2Qwen2_5, WanVAE
    from models.misc import get_text_tokenizer
    from datasets.utils import image_transform
    from utils import get_hyper_params

    device, dtype = torch.device("cuda:0"), torch.bfloat16
    cfg = OmegaConf.load(source / "configs" / "showo2_1.5b_demo_432x432.yaml")
    tokenizer, token_ids = get_text_tokenizer(
        str(args.work / "models" / "qwen2.5-1.5b-tokenizer"),
        add_showo_tokens=True, return_showo_token_ids=True, llm_name="qwen2_5")
    model = Showo2Qwen2_5.from_pretrained(
        str(args.work / "models" / "show-o2-1.5B-local"),
        use_safetensors=False).to(device=device, dtype=dtype).eval()
    vae = WanVAE(vae_pth="/home/shy_23/models/show-o2-deps/Wan2.1_VAE.pth",
                 dtype=dtype, device=device)
    if cfg.model.showo.add_time_embeds:
        cfg.dataset.preprocessing.num_t2i_image_tokens += 1
        cfg.dataset.preprocessing.num_mmu_image_tokens += 1
        cfg.dataset.preprocessing.num_video_tokens += 1
    params = get_hyper_params(cfg, tokenizer, token_ids)
    nvisual = params[1]
    tasks = json.loads(args.tasks_file.read_text(encoding="utf-8"))
    if len(tasks) < 2:
        raise ValueError("Need at least two prompts for negative pairs")
    rng = random.Random(20261003)
    offsets = list(range(len(tasks)))
    rng.shuffle(offsets)
    if any(i == offsets[i] for i in range(len(tasks))):
        offsets = [(i + 1) % len(tasks) for i in range(len(tasks))]
    rows = []
    with torch.inference_mode():
        for i, task in enumerate(tasks):
            seed = task["seeds"][0]
            paths = {
                "base": args.baseline_dir / f"{task['id']}_seed{seed}.png",
                "loop": args.candidate_dir / f"{task['id']}_seed{seed}_L2.png",
            }
            for variant, path in paths.items():
                image = Image.open(path).convert("RGB")
                pixel = image_transform(
                    image, resolution=cfg.dataset.preprocessing.resolution
                ).to(device=device, dtype=dtype)[None]
                latent = vae.sample(pixel.unsqueeze(2)).squeeze(2).to(dtype)
                und = model.image_embedder_und(latent)
                gen = model.image_embedder_gen(latent)
                und = und + model.position_embedding(model.image_position_ids)
                und = model.und_trans(und)["last_hidden_state"]
                visual = model.fusion_proj(torch.cat((und, gen), dim=-1))
                for label, prompt in (("matching", task["prompt"]),
                                      ("mismatched", tasks[offsets[i]]["prompt"])):
                    question = ("Does the image accurately depict this description: "
                                f"{prompt} Answer only yes or no.")
                    response = ask(model, tokenizer, token_ids, visual,
                                   question, nvisual, device, dtype)
                    row = {"id": task["id"], "seed": seed,
                           "variant": variant, "label": label,
                           "response": response, "yes": classify(response)}
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    out = args.work / "stage2b" / f"{args.tag}_mmu_self_consistency.jsonl"
    out.write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                   encoding="utf-8")
    summary = {}
    for variant in ("base", "loop"):
        positive = [r["yes"] for r in rows if r["variant"] == variant
                    and r["label"] == "matching" and r["yes"] is not None]
        negative = [r["yes"] for r in rows if r["variant"] == variant
                    and r["label"] == "mismatched" and r["yes"] is not None]
        summary[variant] = {"positive_yes_rate": sum(positive)/len(positive),
                            "negative_yes_rate": sum(negative)/len(negative),
                            "discrimination_gap": (sum(positive)/len(positive)
                                                   - sum(negative)/len(negative)),
                            "parsed_positive": len(positive),
                            "parsed_negative": len(negative)}
    (args.work / "stage2b" / f"{args.tag}_mmu_self_consistency_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
