"""User-clustered uncertainty analysis for the MGCN behavior audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "audits" / "mgcn_behavior_audit.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "audits" / "mgcn_behavior_audit_statistics.json"
)
CONDITIONS = ("image_zero", "text_zero", "both_zero")
METRICS = ("recall", "ndcg")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def user_metrics(frame: pd.DataFrame, rank_column: str, topk: int) -> pd.DataFrame:
    working = frame[["user_id", rank_column]].copy()
    working["hit"] = working[rank_column] <= topk
    working["gain"] = np.where(
        working["hit"], 1.0 / np.log2(working[rank_column] + 1.0), 0.0
    )
    grouped = working.groupby("user_id", sort=True)
    result = grouped.agg(
        recall=("hit", "mean"),
        dcg=("gain", "sum"),
        positives=("hit", "size"),
    )
    result["idcg"] = result["positives"].map(
        lambda count: float(
            np.sum(1.0 / np.log2(np.arange(2, min(int(count), topk) + 2)))
        )
    )
    result["ndcg"] = result["dcg"] / result["idcg"]
    return result[["recall", "ndcg"]]


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    replications: int,
    chunk_size: int = 250,
) -> tuple[float, float, np.ndarray]:
    means = np.empty(replications, dtype=np.float64)
    count = len(values)
    for start in range(0, replications, chunk_size):
        size = min(chunk_size, replications - start)
        indices = rng.integers(0, count, size=(size, count))
        means[start : start + size] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high), means


def paired_sign_flip_pvalue(
    differences: np.ndarray,
    rng: np.random.Generator,
    replications: int,
    chunk_size: int = 250,
) -> float:
    observed = abs(float(differences.mean()))
    extreme = 0
    count = len(differences)
    for start in range(0, replications, chunk_size):
        size = min(chunk_size, replications - start)
        signs = rng.integers(0, 2, size=(size, count), dtype=np.int8)
        signs = signs * 2 - 1
        permuted = (signs * differences).mean(axis=1)
        extreme += int(np.count_nonzero(np.abs(permuted) >= observed))
    return float((extreme + 1) / (replications + 1))


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for index, (name, pvalue) in enumerate(ordered):
        candidate = min(1.0, (total - index) * pvalue)
        running_max = max(running_max, candidate)
        adjusted[name] = running_max
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--bootstrap-replications", type=int, default=10_000)
    parser.add_argument("--permutation-replications", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    baseline = user_metrics(frame, "baseline_rank", args.topk)
    condition_metrics = {
        condition: user_metrics(frame, f"{condition}_rank", args.topk)
        for condition in CONDITIONS
    }
    if not all(metrics.index.equals(baseline.index) for metrics in condition_metrics.values()):
        raise ValueError("User sets differ between audit conditions")

    rng = np.random.default_rng(args.seed)
    findings: dict[str, dict[str, Any]] = {}
    raw_pvalues: dict[str, float] = {}
    for condition in CONDITIONS:
        for metric in METRICS:
            name = f"{condition}.{metric}@{args.topk}"
            baseline_values = baseline[metric].to_numpy(dtype=np.float64)
            changed_values = condition_metrics[condition][metric].to_numpy(
                dtype=np.float64
            )
            differences = changed_values - baseline_values
            ci_low, ci_high, bootstrap_means = bootstrap_mean_ci(
                differences, rng, args.bootstrap_replications
            )
            raw_pvalue = paired_sign_flip_pvalue(
                differences, rng, args.permutation_replications
            )
            raw_pvalues[name] = raw_pvalue
            difference_std = float(differences.std(ddof=1))
            baseline_mean = float(baseline_values.mean())
            mean_difference = float(differences.mean())
            findings[name] = {
                "condition": condition,
                "metric": f"{metric}@{args.topk}",
                "users": int(len(differences)),
                "baseline_mean": baseline_mean,
                "intervention_mean": float(changed_values.mean()),
                "mean_difference": mean_difference,
                "relative_difference": (
                    mean_difference / baseline_mean if baseline_mean else None
                ),
                "paired_standardized_effect_dz": (
                    mean_difference / difference_std if difference_std else 0.0
                ),
                "bootstrap_95_ci": [ci_low, ci_high],
                "bootstrap_standard_error": float(bootstrap_means.std(ddof=1)),
                "sign_flip_p_raw": raw_pvalue,
            }

    adjusted = holm_adjust(raw_pvalues)
    for name, finding in findings.items():
        finding["sign_flip_p_holm"] = adjusted[name]
        finding["ci_excludes_zero"] = not (
            finding["bootstrap_95_ci"][0] <= 0 <= finding["bootstrap_95_ci"][1]
        )
        finding["holm_below_0_05"] = adjusted[name] < 0.05

    result = {
        "status": "PASSED",
        "input": {
            "path": str(args.input.resolve()),
            "sha256": sha256_file(args.input),
            "rows": int(len(frame)),
            "users": int(frame["user_id"].nunique()),
        },
        "protocol": {
            "analysis_unit": "user",
            "paired": True,
            "topk": args.topk,
            "bootstrap": {
                "type": "user-cluster percentile bootstrap",
                "replications": args.bootstrap_replications,
            },
            "hypothesis_test": {
                "type": "paired user-level sign-flip randomization",
                "replications": args.permutation_replications,
                "two_sided": True,
            },
            "multiple_comparisons": {
                "method": "Holm family-wise error correction",
                "tests": len(findings),
            },
            "seed": args.seed,
        },
        "findings": findings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
