"""Score matched GenEval image pairs with one loaded HPSv2 model."""

import argparse
import json
from pathlib import Path

import hpsv2.img_score as hps
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--baseline-layout", type=Path)
    parser.add_argument("--candidate-layout", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--candidate-suffix", default="_L2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.baseline_layout and args.candidate_layout) == bool(args.baseline_dir and args.candidate_dir):
        parser.error("Specify either both layouts or both image directories")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The complete HPS checkpoint is loaded below. Avoid fetching redundant
    # LAION initialization weights when constructing the backbone.
    model, _, preprocess = hps.create_model_and_transforms(
        "ViT-H-14", pretrained=None, precision="amp", device=str(device),
        jit=False, force_quick_gelu=False, force_custom_text=False,
        force_patch_dropout=False, force_image_size=None, pretrained_image=False,
        image_mean=None, image_std=None, light_augmentation=True, aug_cfg={},
        output_dict=True, with_score_predictor=False,
        with_region_predictor=False,
    )
    checkpoint_path = hf_hub_download(
        "xswu/HPSv2", "HPS_v2_compressed.pt", local_files_only=True
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()
    tokenizer = hps.get_tokenizer("ViT-H-14")

    rows = []
    for task in json.loads(args.tasks.read_text()):
        task_id = task["id"]
        prompt = task["prompt"]
        for seed in task["seeds"]:
            values = []
            with torch.inference_mode():
                tokenized = tokenizer([prompt]).to(device)
                if args.baseline_layout:
                    paths = (args.baseline_layout / task_id / "samples" / "0000.png",
                             args.candidate_layout / task_id / "samples" / "0000.png")
                else:
                    paths = (args.baseline_dir / f"{task_id}_seed{seed}.png",
                             args.candidate_dir / f"{task_id}_seed{seed}{args.candidate_suffix}.png")
                for path in paths:
                    image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
                    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                        output = model(image, tokenized)
                        score = output["image_features"] @ output["text_features"].T
                    values.append(float(score[0, 0].float().cpu()))
            rows.append({"id": task_id, "seed": seed, "category": task["category"], "prompt": prompt,
                         "baseline": values[0], "candidate": values[1],
                         "delta": values[1] - values[0]})
    deltas = np.asarray([row["delta"] for row in rows])
    rng = np.random.default_rng(20261004)
    means = rng.choice(deltas, size=(10000, len(deltas)), replace=True).mean(axis=1)
    summary = {"n": len(rows), "baseline_mean": float(np.mean([r["baseline"] for r in rows])),
               "candidate_mean": float(np.mean([r["candidate"] for r in rows])),
               "delta_mean": float(deltas.mean()),
               "delta_95pct_bootstrap_ci": [float(v) for v in np.percentile(means, [2.5, 97.5])],
               "candidate_wins": int((deltas > 0).sum()), "rows": rows}
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
