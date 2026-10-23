"""Reserve an untouched, category-balanced PartiPrompts generation test set."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def digest(prompt):
    return hashlib.sha256(("stage2b-test-20261003:" + prompt).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=5)
    args = parser.parse_args()
    existing_manifest = json.loads(args.existing.read_text(encoding="utf-8"))
    excluded = {job["prompt"].casefold() for split in ("train", "val")
                for job in existing_manifest[split]}
    grouped = defaultdict(list)
    lines = args.input.read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t") == ["Prompt", "Category", "Challenge", "Note"]
    for line in lines[1:]:
        prompt, category, challenge, _note = line.split("\t", 3)
        prompt = prompt.strip()
        if prompt and prompt.casefold() not in excluded:
            grouped[category].append((prompt, challenge))
    tasks = []
    for category in sorted(grouped):
        seen = set()
        count = 0
        for prompt, challenge in sorted(grouped[category], key=lambda x: digest(x[0])):
            if prompt.casefold() in seen:
                continue
            seen.add(prompt.casefold())
            index = len(tasks)
            tasks.append({"id": f"t{index:04d}", "category": category,
                          "challenge": challenge, "prompt": prompt,
                          "seeds": [20400000 + index * 31]})
            count += 1
            if count == args.per_category:
                break
        if count < args.per_category:
            raise RuntimeError(f"Insufficient unused {category} prompts: {count}")
    args.output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"prompts": len(tasks), "categories": len(grouped)}))


if __name__ == "__main__":
    main()
