"""Clustered statistical analysis across six MGCN intervention conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_mgcn_audit_statistics import (
    bootstrap_mean_ci,
    holm_adjust,
    paired_sign_flip_pvalue,
    sha256_file,
    user_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT / "outputs" / "audits" / "mgcn_intervention_robustness.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "audits"
    / "mgcn_intervention_robustness_statistics.json"
)
CONDITIONS = tuple(
    f"{modality}_{method}"
    for method in ("zero", "mean", "permutation")
    for modality in ("image", "text")
)


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
    rng = np.random.default_rng(args.seed)
    findings: dict[str, dict[str, Any]] = {}
    raw_pvalues: dict[str, float] = {}

    for condition in CONDITIONS:
        changed = user_metrics(frame, f"{condition}_rank", args.topk)
        if not changed.index.equals(baseline.index):
            raise ValueError(f"User mismatch for {condition}")
        for metric in ("recall", "ndcg"):
            name = f"{condition}.{metric}@{args.topk}"
            baseline_values = baseline[metric].to_numpy(dtype=np.float64)
            changed_values = changed[metric].to_numpy(dtype=np.float64)
            differences = changed_values - baseline_values
            low, high, bootstrap_means = bootstrap_mean_ci(
                differences, rng, args.bootstrap_replications
            )
            pvalue = paired_sign_flip_pvalue(
                differences, rng, args.permutation_replications
            )
            raw_pvalues[name] = pvalue
            mean_difference = float(differences.mean())
            baseline_mean = float(baseline_values.mean())
            standard_deviation = float(differences.std(ddof=1))
            findings[name] = {
                "condition": condition,
                "metric": f"{metric}@{args.topk}",
                "baseline_mean": baseline_mean,
                "intervention_mean": float(changed_values.mean()),
                "mean_difference": mean_difference,
                "relative_difference": mean_difference / baseline_mean,
                "paired_standardized_effect_dz": (
                    mean_difference / standard_deviation
                    if standard_deviation
                    else 0.0
                ),
                "bootstrap_95_ci": [low, high],
                "bootstrap_standard_error": float(
                    bootstrap_means.std(ddof=1)
                ),
                "sign_flip_p_raw": pvalue,
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
        "input_sha256": sha256_file(args.input),
        "users": int(frame["user_id"].nunique()),
        "pairs": int(len(frame)),
        "protocol": {
            "analysis_unit": "user",
            "bootstrap_replications": args.bootstrap_replications,
            "sign_flip_replications": args.permutation_replications,
            "multiple_comparisons": "Holm correction across 12 tests",
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
