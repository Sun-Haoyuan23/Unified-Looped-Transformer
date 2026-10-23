"""Run paired whole-head loop exits on every healthy physical GPU."""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def available_gpus():
    listing = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True)
    indices = [int(n) for n in re.findall(r"^GPU (\d+):", listing.stdout, re.MULTILINE)]
    working = []
    for gpu in indices:
        result = subprocess.run(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid,memory.total,memory.used",
             "--format=csv,noheader,nounits"], text=True, capture_output=True
        )
        if result.returncode:
            continue
        uuid, total_str, used_str = [x.strip() for x in result.stdout.strip().split(",")[:3]]
        if int(total_str) - int(used_str) >= 16_000:
            working.append((gpu, uuid))
    return working


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--gpus", help="Comma-separated physical GPU IDs")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--variants", default="L1,L2")
    parser.add_argument("--checkpoint", default="elt_onpolicy_5x2_bf16.pt")
    parser.add_argument("--run-tag", default="elt")
    args = parser.parse_args()
    available = dict(available_gpus())
    gpus = ([(int(g), available[int(g)]) for g in args.gpus.split(",")]
            if args.gpus else list(available.items()))
    if not gpus:
        raise RuntimeError("No GPU has at least 16 GB free memory")
    result_dir = args.work / "stage2b"
    log_dir = result_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print("physical GPUs:", [gpu for gpu, _ in gpus], flush=True)
    children = []
    for shard, (gpu, uuid) in enumerate(gpus):
        env = dict(os.environ)
        env.update({
            "CUDA_VISIBLE_DEVICES": uuid,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "disabled",
            "TOKENIZERS_PARALLELISM": "false",
            "TORCHDYNAMO_DISABLE": "1",
            "PYTHONPATH": str(args.work / "deps") + os.pathsep + env.get("PYTHONPATH", ""),
        })
        cmd = [sys.executable, str(result_dir / "run_elt_onpolicy.py"),
               "--work", str(args.work), "--gpu-physical", str(gpu),
               "--shard", str(shard), "--num-shards", str(len(gpus)),
               "--variants", args.variants,
               "--checkpoint", args.checkpoint,
               "--run-tag", args.run_tag]
        if args.max_images:
            cmd.extend(["--max-images", str(args.max_images)])
        log_path = log_dir / f"{args.run_tag}_worker_gpu{gpu}.log"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        children.append((gpu, proc))
        print("started", gpu, proc.pid, flush=True)
    failures = []
    for gpu, proc in children:
        code = proc.wait()
        print("finished", gpu, code, flush=True)
        if code:
            failures.append(gpu)
    if failures:
        raise SystemExit(f"Failed GPU workers: {failures}")


if __name__ == "__main__":
    main()
