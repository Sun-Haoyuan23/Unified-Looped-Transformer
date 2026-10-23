"""Prepare an isolated, reproducible Show-o2 1.5B inference checkpoint.

The official checkpoint stores all Show-o2 weights. Its config nevertheless
loads a separate Qwen checkpoint while constructing the model. We set
load_from_showo=True to construct Qwen from config, then let Diffusers load the
official complete state dict. A hard link avoids duplicating the 5.66 GB file.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--siglip", type=Path, required=True)
    args = parser.parse_args()

    source = args.work / "models" / "show-o2-1.5B"
    target = args.work / "models" / "show-o2-1.5B-local"
    tokenizer = args.work / "models" / "qwen2.5-1.5b-tokenizer"
    weights = source / "pytorch_model.bin"
    assert weights.stat().st_size > 5_000_000_000, "Show-o2 weight download is incomplete"
    assert (tokenizer / "config.json").is_file()
    assert (args.siglip / "config.json").is_file()

    target.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config["llm_model_path"] = str(tokenizer.resolve())
    config["clip_pretrained_model_path"] = str(args.siglip.resolve())
    config["load_from_showo"] = True
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    linked = target / weights.name
    if linked.exists():
        assert os.path.samefile(weights, linked), "Target weights differ from downloaded checkpoint"
    else:
        os.link(weights, linked)

    manifest = {
        "repository": "showlab/show-o2-1.5B",
        "weight_bytes": weights.stat().st_size,
        "weight_sha256": sha256_file(weights),
        "official_config": str(source / "config.json"),
        "local_config": str(target / "config.json"),
        "config_override": "load_from_showo=True; local Qwen config/tokenizer and SigLIP path",
        "rationale": "full official checkpoint is loaded after construction; strict missing-key check required in smoke run",
    }
    (args.work / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
