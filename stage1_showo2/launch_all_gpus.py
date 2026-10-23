"""Launch one independent Show-o2 baseline shard on every working GPU."""

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
        total, used = int(total_str), int(used_str)
        if total - used >= 16_000:
            working.append((gpu, uuid))
    return working


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--gpus", help="Comma-separated physical indices; default: all working GPUs")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    args = parser.parse_args()
    available = dict(available_gpus())
    gpus = ([(int(g), available[int(g)]) for g in args.gpus.split(",")]
            if args.gpus else list(available.items()))
    if not gpus:
        raise RuntimeError("No GPU with at least 16 GB free memory is available")
    print("physical GPUs:", [gpu for gpu, _ in gpus], flush=True)

    children = []
    for shard, (gpu, uuid) in enumerate(gpus):
        log_path = args.work / "logs" / f"worker_gpu{gpu}.log"
        environment = dict(os.environ)
        environment.update({
            "CUDA_VISIBLE_DEVICES": uuid,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "disabled",
            "TOKENIZERS_PARALLELISM": "false",
            "TORCHDYNAMO_DISABLE": "1",
            "PYTHONPATH": str(args.work / "deps") + os.pathsep + environment.get("PYTHONPATH", ""),
            "TORCHINDUCTOR_CACHE_DIR": str(args.work / "compile_cache" / f"gpu{gpu}"),
        })
        command = [
            sys.executable, str(args.work / "run_baseline.py"),
            "--work", str(args.work), "--gpu-physical", str(gpu),
            "--shard", str(shard), "--num-shards", str(len(gpus)),
            "--steps", str(args.steps), "--guidance", str(args.guidance),
            "--profile-first",
        ]
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        children.append((gpu, process, log_path))
        print("started", gpu, process.pid, log_path, flush=True)

    failures = []
    for gpu, process, log_path in children:
        returncode = process.wait()
        print("finished", gpu, "exit", returncode, log_path, flush=True)
        if returncode:
            failures.append(gpu)
    if failures:
        raise SystemExit(f"Workers failed on physical GPUs: {failures}")


if __name__ == "__main__":
    main()
