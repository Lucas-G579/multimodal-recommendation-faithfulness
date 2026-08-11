"""Compare Kimi prompt v1 and v2 on the isolated development set."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = PROJECT_ROOT / "data" / "manifests" / "llm_evaluation_samples.csv"
V1 = PROJECT_ROOT / "outputs" / "llm_prompt_development" / "kimi_responses.jsonl"
V2 = PROJECT_ROOT / "outputs" / "llm_prompt_development_v2" / "kimi_responses.jsonl"
OUTPUT = PROJECT_ROOT / "data" / "manifests" / "kimi_prompt_v2_comparison.json"
BOOTSTRAP_SEED = 20260811
BOOTSTRAP_ITERATIONS = 10_000


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_attempts(path: Path) -> tuple[list[dict], dict[str, dict], dict[str, list[dict]]]:
    attempts = [json.loads(line) for line in path.open(encoding="utf-8")]
    latest: dict[str, dict] = {}
    histories: dict[str, list[dict]] = {}
    for record in attempts:
        latest[record["sample_id"]] = record
        histories.setdefault(record["sample_id"], []).append(record)
    return attempts, latest, histories


def build_frame(samples: pd.DataFrame, latest: dict[str, dict], prefix: str) -> pd.DataFrame:
    rows = []
    for sample in samples.itertuples(index=False):
        record = latest[sample.sample_id]
        parsed = record.get("parsed") or {}
        prediction = parsed.get("primary_evidence", "invalid")
        effective_status = record["status"]
        if (
            effective_status == "api_error"
            and record.get("error_type") == "ResponseValidationError"
        ):
            effective_status = "schema_invalid"
        valid = effective_status == "valid"
        rows.append(
            {
                "sample_id": sample.sample_id,
                "user_id": int(sample.user_id),
                "label": sample.cross_seed_A_label,
                "contrast": float(sample.contrast_median),
                f"{prefix}_status": effective_status,
                f"{prefix}_prediction": prediction,
                f"{prefix}_text_share": parsed.get("claimed_text_share", np.nan),
                f"{prefix}_confidence": parsed.get("confidence", np.nan),
                f"{prefix}_correct": bool(valid and prediction == sample.cross_seed_A_label),
            }
        )
    return pd.DataFrame(rows)


def macro_recall(frame: pd.DataFrame, correct_column: str) -> float:
    return float(np.mean([
        frame.loc[frame["label"] == label, correct_column].mean()
        for label in ("text", "image")
    ]))


def percentile_interval(values: list[float]) -> list[float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        raise ValueError("No finite bootstrap values")
    return [float(value) for value in np.percentile(finite, [2.5, 97.5])]


def main() -> None:
    samples = pd.read_csv(SAMPLES)
    samples = samples.loc[samples["cohort"] == "prompt_development"].copy()
    if len(samples) != 80:
        raise ValueError("Expected exactly 80 development samples")
    v1_attempts, v1_latest, v1_histories = load_attempts(V1)
    v2_attempts, v2_latest, v2_histories = load_attempts(V2)
    expected_ids = set(samples["sample_id"])
    if set(v1_latest) != expected_ids or set(v2_latest) != expected_ids:
        raise ValueError("v1/v2 responses must exactly cover development IDs")

    v1 = build_frame(samples, v1_latest, "v1")
    v2 = build_frame(samples, v2_latest, "v2")
    frame = v1.merge(
        v2.drop(columns=["user_id", "label", "contrast"]),
        on="sample_id",
        validate="one_to_one",
    )
    unique_users = np.array(sorted(frame["user_id"].unique()))
    groups = {user: frame.index[frame["user_id"] == user].to_numpy() for user in unique_users}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    v2_accuracy_boot = []
    v2_macro_boot = []
    macro_delta_boot = []
    v2_spearman_boot = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sampled_users = rng.choice(unique_users, size=len(unique_users), replace=True)
        indices = np.concatenate([groups[user] for user in sampled_users])
        sample = frame.loc[indices]
        v1_macro = macro_recall(sample, "v1_correct")
        v2_macro = macro_recall(sample, "v2_correct")
        v2_accuracy_boot.append(float(sample["v2_correct"].mean()))
        v2_macro_boot.append(v2_macro)
        macro_delta_boot.append(v2_macro - v1_macro)
        valid = sample.loc[sample["v2_status"] == "valid"]
        correlation = spearmanr(
            valid["v2_text_share"], valid["contrast"], nan_policy="omit"
        ).statistic
        if np.isfinite(correlation):
            v2_spearman_boot.append(float(correlation))

    v2_valid = frame.loc[frame["v2_status"] == "valid"]
    v1_macro = macro_recall(frame, "v1_correct")
    v2_macro = macro_recall(frame, "v2_correct")
    recalls = {
        label: float(frame.loc[frame["label"] == label, "v2_correct"].mean())
        for label in ("text", "image")
    }
    retry_samples = [values for values in v2_histories.values() if len(values) > 1]
    final_failures = [values[-1] for values in v2_histories.values() if values[-1]["status"] != "valid"]
    effective_final_statuses = []
    for record in v2_latest.values():
        status = record["status"]
        if status == "api_error" and record.get("error_type") == "ResponseValidationError":
            status = "schema_invalid"
        effective_final_statuses.append(status)
    usage_fields = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens")
    summary = {
        "status": "V2_DEVELOPMENT_ANALYZED_AND_FROZEN",
        "scope": {"development_samples": 80, "confirmatory_samples": 0},
        "v2_execution": {
            "attempt_records": len(v2_attempts),
            "final_status_counts": dict(Counter(effective_final_statuses)),
            "retry_samples": len(retry_samples),
            "retry_recovered": sum(v[-1]["status"] == "valid" for v in retry_samples),
            "permanent_failure_types": dict(Counter(r["error_type"] for r in final_failures)),
            "usage_all_attempts": {
                field: sum(int(r.get("usage", {}).get(field, 0) or 0) for r in v2_attempts)
                for field in usage_fields
            },
        },
        "v2_predictions_valid_only": dict(Counter(v2_valid["v2_prediction"])),
        "v2_intention_to_treat": {
            "accuracy": float(frame["v2_correct"].mean()),
            "accuracy_cluster_bootstrap_95ci": percentile_interval(v2_accuracy_boot),
            "text_recall": recalls["text"],
            "image_recall": recalls["image"],
            "macro_recall": v2_macro,
            "macro_recall_cluster_bootstrap_95ci": percentile_interval(v2_macro_boot),
            "high_confidence_wrong_count": int(
                ((~frame["v2_correct"]) & (frame["v2_confidence"] >= 0.8)).sum()
            ),
        },
        "v2_valid_only": {
            "accuracy": float(v2_valid["v2_correct"].mean()),
            "mean_confidence": float(v2_valid["v2_confidence"].mean()),
            "spearman_text_share_vs_contrast": float(
                spearmanr(v2_valid["v2_text_share"], v2_valid["contrast"]).statistic
            ),
            "spearman_cluster_bootstrap_95ci": percentile_interval(v2_spearman_boot),
        },
        "paired_development_comparison": {
            "v1_macro_recall": v1_macro,
            "v2_macro_recall": v2_macro,
            "macro_recall_delta": v2_macro - v1_macro,
            "macro_recall_delta_cluster_bootstrap_95ci": percentile_interval(macro_delta_boot),
        },
        "bootstrap": {
            "unit": "user_id",
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "unique_users": int(len(unique_users)),
        },
        "decision": (
            "Freeze v2 for one-shot confirmatory evaluation; do not create v3 from "
            "further inspection of this development set."
        ),
        "inputs": {
            "samples_sha256": sha256_file(SAMPLES),
            "v1_responses_sha256": sha256_file(V1),
            "v2_responses_sha256": sha256_file(V2),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
