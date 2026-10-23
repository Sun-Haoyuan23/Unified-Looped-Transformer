"""Shard student-rollout teacher queries over usable GPUs by UUID."""

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
            ["nvidia-smi", "-i", str(gpu),
             "--query-gpu=uuid,memory.total,memory.used",
             "--format=csv,noheader,nounits"], text=True, capture_output=True)
        if result.returncode:
            continue
        uuid, total, used = [x.strip() for x in result.stdout.strip().split(",")[:3]]
        if int(total) - int(used) >= 16000:
            working.append((gpu, uuid))
    return working


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--gpus", help="Physical GPU IDs, comma-separated")
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--loops", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    available = dict(available_gpus())
    gpus = ([(int(g), available[int(g)]) for g in args.gpus.split(",")]
            if args.gpus else list(available.items()))
    if not gpus:
        raise RuntimeError("No GPU has 16GB available")
    log_dir = args.work / "stage2b" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    children = []
    print("physical GPUs", [g for g, _ in gpus], flush=True)
    for shard, (gpu, uuid) in enumerate(gpus):
        env = dict(os.environ)
        env.update({"CUDA_VISIBLE_DEVICES": uuid, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                    "WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false",
                    "TORCHDYNAMO_DISABLE": "1",
                    "PYTHONPATH": str(args.work / "deps") + os.pathsep + env.get("PYTHONPATH", "")})
        cmd = [sys.executable, str(args.work / "stage2b" / "collect_student_rollouts.py"),
               "--work", str(args.work), "--manifest", str(args.manifest),
               "--split", args.split, "--gpu-physical", str(gpu),
               "--shard", str(shard), "--num-shards", str(len(gpus)),
               "--loops", str(args.loops)]
        if args.max_prompts:
            cmd.extend(["--max-prompts", str(args.max_prompts)])
        with (log_dir / f"collect_student_L{args.loops}_{args.split}_worker_gpu{gpu}.log").open("w") as log:
            process = subprocess.Popen(cmd, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        children.append((gpu, process))
        print("started", gpu, process.pid, flush=True)
    failures = []
    for gpu, process in children:
        code = process.wait()
        print("finished", gpu, code, flush=True)
        if code:
            failures.append(gpu)
    if failures:
        raise SystemExit(f"Worker failures: {failures}")


if __name__ == "__main__":
    main()
