"""Offline, original-prompt preference/alignment scoring and strict score merging.

Public stdlib APIs (no Torch/model imports until score_candidates is called):
  load_action_records(paths, baseline_action='baseline') -> list[dict]
  prepare_score_rows(records) -> list[dict]  # validates files and hashes images
  merge_score_rows(records, score_streams, *, require_metrics=()) -> list[dict]
  score_candidates(records, *, work, metrics, device='cuda:0', batch_size=4,
                   hps_checkpoint=None) -> list[dict]

Scoring:
  python score_repair_candidates.py --records candidates/actions_shard*.jsonl
    --work /home/user/work --metrics pickscore --output pick.jsonl
  python score_repair_candidates.py --records candidates --work /home/user/work
    --metrics hps,siglip --device cuda:0 --batch-size 4 --output hps_siglip.jsonl
Merging (stdlib-only, no model loading):
  python score_repair_candidates.py --records candidates --merge pick.jsonl
    hps_siglip.jsonl --require-metrics pickscore,hps,siglip --output scores.jsonl

A --records directory recursively selects only actions_shard*.jsonl, including
experiment/shard*/actions_shard0.jsonl. It never reads all_actions.jsonl or score
outputs implicitly. Duplicate
keys, incomplete state/action sets, missing images or original-prompt mismatch
fail. Every input metric file in --merge must cover exactly the entire action key
set; uncomputed scores must be explicit null (not absent rows). A metric supplied
twice for a key is rejected, even if values match. --require-metrics is optional;
otherwise genuinely uncomputed metrics stay null. Hashes must match the current
image bytes and original prompt. No models are downloaded: Transformers loads
only local paths and HPS loads only an existing checkpoint/cache entry.

Definitions match the preceding stage2 evaluators:
  PickScore: exp(logit_scale) * normalized image/text feature cosine (raw logit).
  HPSv2: normalized ViT-H-14 image/text feature dot product, without logit scale.
  SigLIP: normalized image/text cosine, without sigmoid/logit scale or bias.
Text features are cached by original prompt; prompt_repair is always scored
against baseline's ORIGINAL prompt. Metrics load sequentially and release CUDA
memory between models. HPS is a separate scoring process; no Show-o2 is loaded.
These are automatic proxies, not human preference votes or diversity measures.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import glob
import hashlib
import json
import math
from pathlib import Path


METRICS = ("pickscore", "hps", "siglip")
SCORE_DEFINITIONS = {
    "pickscore": "exp(logit_scale) * normalized image/text cosine; raw PickScore logit",
    "hps": "normalized ViT-H-14 image/text dot product; HPS_v2_compressed.pt; no logit scale",
    "siglip": "normalized SigLIP image/text cosine; no logit scale, bias or sigmoid",
}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL objects required: {path}")
    return rows


def key(row):
    value = row.get("state_id"), row.get("action")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError("Each row requires nonempty string state_id and action")
    return value


def _index(rows, label):
    lookup = {}
    for row in rows:
        item = key(row)
        if item in lookup:
            raise ValueError(f"Duplicate {label} key: {item}")
        lookup[item] = row
    return lookup


def expand_record_paths(paths):
    selected = []
    for item in paths:
        candidate = Path(item)
        if candidate.is_dir():
            matches = sorted(path for path in candidate.rglob("actions_shard*.jsonl") if path.is_file())
        else:
            matches = [Path(match) for match in sorted(glob.glob(str(item), recursive=True)) if Path(match).is_file()]
        if not matches:
            raise FileNotFoundError(f"No action records match: {item}")
        selected.extend(matches)
    resolved = [path.resolve() for path in selected]
    if len(resolved) != len(set(resolved)):
        raise ValueError("The same action-record file was supplied more than once")
    return resolved


def load_action_records(paths, baseline_action="baseline"):
    rows = []
    for path in expand_record_paths(paths):
        rows.extend(read_jsonl(path))
    return _validate_records(rows, baseline_action)


def _validate_records(records, baseline_action="baseline"):
    lookup = _index(records, "action records")
    if not lookup:
        raise ValueError("No action records")
    states = defaultdict(dict)
    for (state, action), row in lookup.items():
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"Original prompt required at {(state, action)}")
        if not isinstance(row.get("image_path"), str) or not row["image_path"]:
            raise ValueError(f"image_path required at {(state, action)}")
        states[state][action] = row
    expected = None
    output = []
    for state, group in sorted(states.items()):
        if baseline_action not in group or len(group) < 2:
            raise ValueError(f"Each state needs baseline and a candidate: {state}")
        if expected is None:
            expected = set(group)
        elif set(group) != expected:
            raise ValueError(f"Incomplete action set at {state}")
        baseline = group[baseline_action]
        # The original prompt must not be replaced by a prompt-repair override.
        original = baseline.get("original_prompt", baseline["prompt"])
        if not isinstance(original, str) or not original.strip():
            raise ValueError("Baseline original_prompt must be a nonempty string")
        for action, row in sorted(group.items()):
            if row.get("original_prompt", row["prompt"]) != original or row["prompt"] != original:
                raise ValueError(f"Original scoring prompt mismatch at {(state, action)}; store generation override in a separate field")
            for field in ("prompt_id", "seed", "flow_index", "true_t", "split", "state_fingerprint"):
                if row.get(field) != baseline.get(field):
                    raise ValueError(f"Same-state {field} mismatch at {(state, action)}")
            output.append(dict(row, scored_prompt=original))
    return output


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_score_rows(records):
    records = _validate_records(records)
    rows = []
    for record in records:
        path = Path(record["image_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Candidate image missing: {path}")
        original = record["scored_prompt"]
        rows.append({"state_id": record["state_id"], "action": record["action"],
                     "prompt_id": record.get("prompt_id"), "seed": record.get("seed"),
                     "flow_index": record.get("flow_index"), "true_t": record.get("true_t"),
                     "image_path": str(path), "image_sha256": file_sha256(path),
                     "scored_prompt": original,
                     "scored_prompt_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
                     "pickscore": None, "hps": None, "siglip": None,
                     "metric_models": {}, "score_definitions": {}})
    return rows


def _score(value, label):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number or explicit null")
    return float(value)


def _metrics(values):
    values = tuple(values)
    if any(value not in METRICS for value in values) or len(values) != len(set(values)):
        raise ValueError(f"Metrics must be unique names from {METRICS}")
    return values


def merge_score_rows(records, score_streams, *, require_metrics=()):
    required = _metrics(require_metrics)
    expected_rows = prepare_score_rows(records)
    expected = _index(expected_rows, "expected records")
    streams = list(score_streams)
    if not streams:
        raise ValueError("At least one score stream required for merge")
    for stream_index, stream in enumerate(streams):
        lookup = _index(stream, f"score stream {stream_index}")
        if set(lookup) != set(expected):
            raise ValueError(f"Score stream {stream_index} has missing or extra action rows; exact key alignment required")
        for item, partial in lookup.items():
            row = expected[item]
            for field in ("image_sha256", "scored_prompt_sha256"):
                if partial.get(field) != row[field]:
                    raise ValueError(f"{field} mismatch at {item}; image or original prompt changed")
            if partial.get("scored_prompt") != row["scored_prompt"]:
                raise ValueError(f"Original scored prompt mismatch at {item}")
            if partial.get("prompt_id") != row["prompt_id"]:
                raise ValueError(f"prompt_id mismatch at {item}")
            for metric in METRICS:
                if metric not in partial:
                    raise ValueError(f"Explicit {metric}:null required for uncomputed metrics at {item}")
                value = _score(partial[metric], metric)
                if value is not None:
                    if row[metric] is not None:
                        raise ValueError(f"Duplicate nonnull metric {metric} at {item}")
                    row[metric] = value
                    if metric in partial.get("metric_models", {}):
                        row["metric_models"][metric] = partial["metric_models"][metric]
                    if metric in partial.get("score_definitions", {}):
                        row["score_definitions"][metric] = partial["score_definitions"][metric]
    for item, row in expected.items():
        missing = [metric for metric in required if row[metric] is None]
        if missing:
            raise ValueError(f"Required metrics missing at {item}: {missing}")
    return [expected[item] for item in sorted(expected)]


def _open_images(rows):
    from PIL import Image
    output = []
    for row in rows:
        with Image.open(row["image_path"]) as image:
            output.append(image.convert("RGB"))
    return output


def _feature_tensor(value):
    # Some Transformers releases return ModelOutput from get_*_features.
    return value.pooler_output if hasattr(value, "pooler_output") else value


def _score_transformers(rows, *, metric, work, device, batch_size):
    import torch
    from transformers import AutoModel, AutoProcessor, SiglipModel
    if metric == "pickscore":
        processor_path = work / "models" / "pickscore-processor"
        model_path = work / "models" / "pickscore-v1"
        model_type = AutoModel
    else:
        processor_path = model_path = work / "models" / "siglip-so400m"
        model_type = SiglipModel
    if not model_path.is_dir() or not processor_path.is_dir():
        raise FileNotFoundError(f"Local {metric} model/processor unavailable: {model_path}, {processor_path}")
    processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
    load_kwargs = {"local_files_only": True}
    if metric == "pickscore":
        load_kwargs["use_safetensors"] = True
    model = model_type.from_pretrained(model_path, **load_kwargs).eval().to(device)
    text_features = {}
    with torch.inference_mode():
        for prompt in dict.fromkeys(row["scored_prompt"] for row in rows):
            options = {"padding": True, "truncation": True, "max_length": 77} if metric == "pickscore" else {"padding": "max_length"}
            encoded = processor(text=[prompt], return_tensors="pt", **options)
            encoded = {name: value.to(device) for name, value in encoded.items()}
            vector = _feature_tensor(model.get_text_features(**encoded)).float()
            text_features[prompt] = torch.nn.functional.normalize(vector, dim=-1)[0].cpu()
        scale = float(model.logit_scale.exp().detach().cpu()) if metric == "pickscore" else 1.0
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            images = _open_images(batch)
            if metric == "pickscore":
                encoded = processor(images=images, padding=True, truncation=True,
                                    max_length=77, return_tensors="pt")
                encoded = {name: value.to(device) for name, value in encoded.items()}
                vector = _feature_tensor(model.get_image_features(**encoded)).float()
            else:
                encoded = processor(images=images, return_tensors="pt")
                vector = _feature_tensor(model.get_image_features(pixel_values=encoded["pixel_values"].to(device))).float()
            vector = torch.nn.functional.normalize(vector, dim=-1).cpu()
            for row, feature in zip(batch, vector):
                row[metric] = scale * float(torch.dot(text_features[row["scored_prompt"]], feature))
                row["metric_models"][metric] = str(model_path)
                row["score_definitions"][metric] = SCORE_DEFINITIONS[metric]
            print(json.dumps({"metric": metric, "scored": min(offset + batch_size, len(rows)), "total": len(rows)}), flush=True)
            for image in images:
                image.close()
    # Drop GPU tensors in function scope before the next metric is loaded.
    del model, text_features


def _score_hps(rows, *, work, device, batch_size, hps_checkpoint):
    import torch
    import hpsv2.img_score as hps
    from huggingface_hub import hf_hub_download
    if hps_checkpoint is None:
        conventional = work / "models" / "HPS_v2_compressed.pt"
        checkpoint_path = conventional if conventional.is_file() else Path(hf_hub_download(
            "xswu/HPSv2", "HPS_v2_compressed.pt", local_files_only=True))
    else:
        checkpoint_path = Path(hps_checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Local HPS checkpoint missing: {checkpoint_path}")
    model, _, preprocess = hps.create_model_and_transforms(
        "ViT-H-14", pretrained=None, precision="amp", device=str(device),
        jit=False, force_quick_gelu=False, force_custom_text=False,
        force_patch_dropout=False, force_image_size=None, pretrained_image=False,
        image_mean=None, image_std=None, light_augmentation=True, aug_cfg={},
        output_dict=True, with_score_predictor=False, with_region_predictor=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    del checkpoint
    model = model.to(device).eval()
    tokenizer = hps.get_tokenizer("ViT-H-14")
    text_features = {}
    torch_device = torch.device(device)
    autocast_enabled = torch_device.type == "cuda"
    with torch.inference_mode():
        for prompt in dict.fromkeys(row["scored_prompt"] for row in rows):
            tokenized = tokenizer([prompt]).to(torch_device)
            with torch.autocast(device_type=torch_device.type, enabled=autocast_enabled):
                vector = model.encode_text(tokenized, normalize=True)
            text_features[prompt] = vector[0].float().cpu()
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            images = _open_images(batch)
            image_tensor = torch.stack([preprocess(image) for image in images]).to(torch_device)
            with torch.autocast(device_type=torch_device.type, enabled=autocast_enabled):
                image_vectors = model.encode_image(image_tensor, normalize=True)
                # Match old HPS autocast forward: dot product in autocast precision.
                prompt_vectors = torch.stack([text_features[row["scored_prompt"]] for row in batch]).to(
                    device=torch_device, dtype=image_vectors.dtype)
                similarity = image_vectors @ prompt_vectors.T
            values = similarity.diagonal().float().cpu().tolist()
            for row, value in zip(batch, values):
                row["hps"] = float(value)
                row["metric_models"]["hps"] = str(checkpoint_path)
                row["score_definitions"]["hps"] = SCORE_DEFINITIONS["hps"]
            print(json.dumps({"metric": "hps", "scored": min(offset + batch_size, len(rows)), "total": len(rows)}), flush=True)
            for image in images:
                image.close()
    del model, text_features


def score_candidates(records, *, work, metrics, device="cuda:0", batch_size=4, hps_checkpoint=None):
    selected = _metrics(metrics)
    if not selected or batch_size < 1:
        raise ValueError("At least one metric and a positive batch_size required")
    work = Path(work)
    rows = prepare_score_rows(records)
    for metric in selected:
        if metric == "hps":
            _score_hps(rows, work=work, device=device, batch_size=batch_size, hps_checkpoint=hps_checkpoint)
        else:
            _score_transformers(rows, metric=metric, work=work, device=device, batch_size=batch_size)
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for row in rows:
        if file_sha256(row["image_path"]) != row["image_sha256"]:
            raise RuntimeError(f"Image changed during scoring: {row['image_path']}")
        for metric in METRICS:
            _score(row[metric], metric)
            if metric in selected and row[metric] is None:
                raise RuntimeError(f"Scoring failed to produce requested {metric} at {key(row)}")
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--metrics", default="pickscore,hps,siglip")
    parser.add_argument("--merge", type=Path, nargs="+")
    parser.add_argument("--require-metrics", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hps-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    records = load_action_records(args.records)
    if args.merge:
        rows = merge_score_rows(records, [read_jsonl(path) for path in args.merge],
                                require_metrics=tuple(filter(None, args.require_metrics.split(","))))
    else:
        if args.work is None:
            parser.error("--work is required for model scoring (not for --merge)")
        if args.require_metrics:
            parser.error("--require-metrics is a merge validation option")
        rows = score_candidates(records, work=args.work, metrics=tuple(args.metrics.split(",")),
                                device=args.device, batch_size=args.batch_size, hps_checkpoint=args.hps_checkpoint)
    write_jsonl(args.output, rows)
    print(json.dumps({"output": str(args.output), "rows": len(rows),
                      "mode": "merge" if args.merge else "score",
                      "nonnull_metric_counts": {metric: sum(row[metric] is not None for row in rows) for metric in METRICS},
                      "scoring_prompt": "original baseline prompt"}), flush=True)


if __name__ == "__main__":
    main()
