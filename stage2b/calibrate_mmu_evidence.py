"""Test whether Show-o2's own image understanding predicts GenEval failures.

This is a gate prerequisite, not a generation or understanding benchmark.
It uses final images because clean-image estimates at intermediate flow times
would add a second source of error. The latter must be calibrated separately.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

def auc(scores, labels):
    positive = [s for s, y in zip(scores, labels) if y]
    negative = [s for s, y in zip(scores, labels) if not y]
    if not positive or not negative:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in positive for b in negative) / (len(positive) * len(negative))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--layout", type=Path)
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--requested-t", type=float)
    parser.add_argument("--official-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--question-mode", choices=("prompt", "relation_counterfactual"),
                        default="prompt")
    args = parser.parse_args()
    if bool(args.layout) == bool(args.trajectory_dir):
        parser.error("Specify either --layout or --trajectory-dir")
    if args.trajectory_dir and args.requested_t is None:
        parser.error("--trajectory-dir requires --requested-t")
    source = args.work / "Show-o" / "show-o2"
    sys.path.insert(0, str(source))
    from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
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
    vae = (WanVAE(vae_pth="/home/shy_23/models/show-o2-deps/Wan2.1_VAE.pth",
                  dtype=dtype, device=device) if args.layout else None)
    if cfg.model.showo.add_time_embeds:
        cfg.dataset.preprocessing.num_t2i_image_tokens += 1
        cfg.dataset.preprocessing.num_mmu_image_tokens += 1
        cfg.dataset.preprocessing.num_video_tokens += 1
    nvisual = get_hyper_params(cfg, tokenizer, token_ids)[1]
    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    tasks = [task for i, task in enumerate(tasks) if i % args.num_shards == args.shard]
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    official = {}
    official_meta = {}
    for line in args.official_results.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        task_id = Path(row["filename"]).parent.parent.name
        official[task_id] = bool(row["correct"])
        official_meta[task_id] = json.loads(row["metadata"])
    yes_ids = {tokenizer(t, add_special_tokens=False).input_ids[0]
               for t in ("Yes", " yes", "YES", "yes") if tokenizer(t, add_special_tokens=False).input_ids}
    no_ids = {tokenizer(t, add_special_tokens=False).input_ids[0]
              for t in ("No", " no", "NO", "no") if tokenizer(t, add_special_tokens=False).input_ids}
    # Remove any accidental collision from tokenizer variants.
    yes_ids -= no_ids
    print(json.dumps({"yes_ids": sorted(yes_ids), "no_ids": sorted(no_ids)}), flush=True)

    def question_margin(visual, question):
        system = tokenizer("system\nYou are a helpful assistant.<|im_end|>", add_special_tokens=False).input_ids
        user = tokenizer("\n<|im_start|>user\n", add_special_tokens=False).input_ids
        assistant = tokenizer("\n<|im_start|>assistant\n", add_special_tokens=False).input_ids
        query = tokenizer(question, add_special_tokens=False).input_ids
        a = torch.tensor([token_ids["bos_id"]] + system + user, device=device)[None]
        b = torch.tensor([token_ids["boi_id"], token_ids["eoi_id"]] + query + assistant, device=device)[None]
        ae = model.showo.model.embed_tokens(a)
        be = model.showo.model.embed_tokens(b)
        time = model.time_embed(torch.tensor([[1.0]], device=device), ae.dtype)
        if hasattr(model, "time_embed_proj"):
            time = model.time_embed_proj(time)
        inputs = torch.cat((ae, be[:, :1], time, visual, be[:, 1:]), dim=1).to(dtype)
        positions = torch.tensor([a.shape[1] + 2, nvisual], device=device)[None, None]
        mask = omni_attn_mask_naive(B=1, LEN=inputs.shape[1], modalities=positions,
                                    device=device, inverted=True).to(inputs.dtype)
        logits = model.showo(inputs_embeds=inputs, attention_mask=mask)["logits"][0, -1].float()
        margin = torch.logsumexp(logits[list(yes_ids)], dim=0) - torch.logsumexp(logits[list(no_ids)], dim=0)
        return float(margin)

    rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for task in tasks:
            task_id = task["id"]
            if args.question_mode == "relation_counterfactual" and task["category"] != "position":
                continue
            if args.trajectory_dir:
                payload = torch.load(args.trajectory_dir / f"{task_id}.pt",
                                     map_location="cpu", weights_only=False)
                times = payload["t"].float()
                index = int(torch.argmin((times - args.requested_t).abs()))
                actual_t = float(times[index])
                latent = (payload["zt"][index].float() +
                          (1 - actual_t) * payload["teacher_velocity"][index].float())
                latent = latent.to(device=device, dtype=dtype)
            else:
                image = Image.open(args.layout / task_id / "samples" / "0000.png").convert("RGB")
                pixels = image_transform(image, resolution=cfg.dataset.preprocessing.resolution)
                pixels = pixels.to(device=device, dtype=dtype)[None]
                torch.manual_seed(1234 + int(task_id))
                latent = vae.sample(pixels.unsqueeze(2)).squeeze(2).to(dtype)
            und = model.image_embedder_und(latent)
            gen = model.image_embedder_gen(latent)
            und = und + model.position_embedding(model.image_position_ids)
            und = model.und_trans(und)["last_hidden_state"]
            visual = model.fusion_proj(torch.cat((und, gen), dim=-1))
            if args.question_mode == "prompt":
                question = ("Does the image accurately depict this description: "
                            f"{task['prompt']} Answer only Yes or No.")
                margin = question_margin(visual, question)
                extra = {}
            else:
                included = official_meta[task_id]["include"]
                positioned = next(item for item in included if "position" in item)
                relation, reference_index = positioned["position"]
                reference = included[reference_index]["class"]
                target = positioned["class"]
                opposites = {"above": "below", "below": "above",
                             "left of": "right of", "right of": "left of"}
                preferred = question_margin(
                    visual, f"In this image, is the {target} {relation} the {reference}? Answer only Yes or No.")
                counterfactual = question_margin(
                    visual, f"In this image, is the {target} {opposites[relation]} the {reference}? Answer only Yes or No.")
                margin = preferred - counterfactual
                extra = {"desired_relation": relation, "preferred_yes_no_margin": preferred,
                         "opposite_yes_no_margin": counterfactual}
            answer_yes = margin > 0
            row = {"id": task_id, "category": task["category"], "prompt": task["prompt"],
                   "official_correct": official[task_id], "yes_no_logit_margin": margin,
                   "model_says_yes": answer_yes, **extra}
            if args.trajectory_dir:
                row["actual_t"] = actual_t
            rows.append(row)
            print(json.dumps(row), flush=True)
    labels = [r["official_correct"] for r in rows]
    scores = [r["yes_no_logit_margin"] for r in rows]
    parsed = [r["model_says_yes"] for r in rows]
    summary = {"n": len(rows), "official_correct": sum(labels),
               "mmu_yes": sum(parsed), "mmu_accuracy_vs_official": sum(a == b for a, b in zip(labels, parsed)) / len(rows),
               "auc": auc(scores, labels), "rows": rows}
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
