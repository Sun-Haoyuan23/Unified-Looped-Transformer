"""Account for a grounded loop's supported-operator inference FLOPs.

The input profiler measures one CFG velocity call at fixed resolution. We
compose that with actual per-image visual-block calls and one direct-latent
MMU check. This is an operator-coverage proxy, not hardware-total FLOPs.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--mmu-profile", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = {row["name"]: row for row in json.loads(args.profile.read_text())}
    baseline = profile["released_10_unique_10_calls"][
        "profiled_operator_flops_per_velocity"]
    short = profile["shared_single_9_unique_9_calls"][
        "profiled_operator_flops_per_velocity"]
    full = profile["shared_single_9_unique_10_calls"][
        "profiled_operator_flops_per_velocity"]
    per_block = full - short
    mmu = json.loads(args.mmu_profile.read_text())["profiled_operator_flops"]
    rows = [json.loads(line) for path in args.metrics
            for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("status") == "ok"]
    if not rows:
        raise ValueError("No completed images")
    results = []
    for row in rows:
        steps = row["flow_velocity_calls"]
        block_calls = row["total_visual_block_calls"]
        if block_calls < 9 * steps:
            raise ValueError(f"Invalid block count: {row['id']}")
        base_flops = steps * baseline
        candidate_flops = steps * short + (block_calls - 9 * steps) * per_block + mmu
        results.append({"id": row["id"], "seed": row["seed"],
                        "triggered": row.get("extra_loop_triggered"),
                        "visual_block_calls": block_calls,
                        "baseline_operator_tflops": base_flops / 1e12,
                        "candidate_operator_tflops": candidate_flops / 1e12,
                        "relative_change_pct": 100 * (candidate_flops / base_flops - 1)})
    report = {
        "n": len(results),
        "unique_parameters_baseline": profile["released_10_unique_10_calls"][
            "total_unique_parameters"],
        "unique_parameters_candidate": profile["shared_single_9_unique_9_calls"][
            "total_unique_parameters"],
        "mean_baseline_operator_tflops": sum(r["baseline_operator_tflops"]
                                             for r in results) / len(results),
        "mean_candidate_operator_tflops": sum(r["candidate_operator_tflops"]
                                              for r in results) / len(results),
        "mean_relative_change_pct": sum(r["relative_change_pct"]
                                        for r in results) / len(results),
        "trigger_rate": sum(bool(r["triggered"]) for r in results) / len(results),
        "note": "PyTorch-supported operators; matched 432px CFG calls. Excludes VAE and unsupported operators.",
        "rows": results,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))


if __name__ == "__main__":
    main()
