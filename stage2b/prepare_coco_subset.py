"""Download a reproducible, disjoint COCO-2017 train-image pilot subset.

Source: official COCO 2017 train JPEG URLs; captions_train2017.json must be
obtained separately. This is a post-training pilot, not the COCO val FID set.
"""

import argparse
import json
import random
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from PIL import Image


def fetch(item, destination):
    path = destination / item["file_name"]
    if path.exists():
        try:
            with Image.open(path) as image:
                image.verify()
            return item["id"], "cached"
        except Exception:
            path.unlink()
    url = f"http://images.cocodataset.org/train2017/{item['file_name']}"
    last_error = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = response.read()
            with Image.open(BytesIO(payload)) as image:
                image.verify()
            temp = path.with_suffix(".part")
            temp.write_bytes(payload)
            temp.replace(path)
            return item["id"], "downloaded"
        except Exception as error:
            last_error = repr(error)
            time.sleep(1 + attempt)
    return item["id"], last_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=1024)
    parser.add_argument("--val-count", type=int, default=128)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.annotations.read_text(encoding="utf-8"))
    captions = defaultdict(list)
    for annotation in source["annotations"]:
        caption = " ".join(annotation["caption"].strip().split())
        if 20 <= len(caption) <= 250:
            captions[annotation["image_id"]].append(caption)
    available = [item for item in source["images"]
                 if item["id"] in captions and
                 min(item["width"], item["height"]) >= 400]
    rng = random.Random(20261003)
    rng.shuffle(available)
    selected = available[:args.train_count + args.val_count]
    if len(selected) != args.train_count + args.val_count:
        raise RuntimeError("Not enough eligible COCO images")
    records = []
    for index, item in enumerate(selected):
        records.append({"id": item["id"], "file_name": item["file_name"],
                        "prompt": max(captions[item["id"]], key=len),
                        "split": "train" if index < args.train_count else "val",
                        "source": "COCO-2017 train images and captions"})
    image_dir = args.output / "images"
    image_dir.mkdir(exist_ok=True)
    counts = defaultdict(int)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(fetch, item, image_dir): item for item in selected}
        for index, future in enumerate(as_completed(jobs), start=1):
            image_id, status = future.result()
            counts[status if status in ("cached", "downloaded") else "failed"] += 1
            if status not in ("cached", "downloaded"):
                failures.append({"id": image_id, "error": status})
            if index % 100 == 0 or index == len(jobs):
                print(json.dumps({"done": index, "total": len(jobs),
                                  "counts": counts, "failures": len(failures)}),
                      flush=True)
    valid = [item for item in records if (image_dir / item["file_name"]).exists()]
    (args.output / "manifest.json").write_text(
        json.dumps({"seed": 20261003, "records": valid,
                    "failures": failures}, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)} images failed to download")


if __name__ == "__main__":
    main()
