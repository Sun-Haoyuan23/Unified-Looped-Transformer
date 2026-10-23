"""Download the official independent PickScore evaluator into the project."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    root = args.work / "models"
    score_dir = root / "pickscore-v1"
    processor_dir = root / "pickscore-processor"
    snapshot_download("yuvalkirstain/PickScore_v1", local_dir=score_dir,
                      allow_patterns=["config.json", "*.safetensors"], max_workers=4)
    snapshot_download("laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                      local_dir=processor_dir,
                      allow_patterns=["*.json", "merges.txt", "vocab.json"],
                      max_workers=4)
    manifest = {"model_repo": "yuvalkirstain/PickScore_v1",
                "processor_repo": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                "model_dir": str(score_dir), "processor_dir": str(processor_dir),
                "model_files": {p.name: p.stat().st_size for p in score_dir.iterdir() if p.is_file()},
                "processor_files": {p.name: p.stat().st_size for p in processor_dir.iterdir() if p.is_file()}}
    (args.work / "stage2_loop" / "pickscore_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
