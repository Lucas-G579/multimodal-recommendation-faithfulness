"""Analyze Kimi prompt-development outputs without touching confirmatory samples."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESPONSES = PROJECT_ROOT / "outputs" / "llm_prompt_development" / "kimi_responses.jsonl"
SAMPLES = PROJECT_ROOT / "data" / "manifests" / "llm_evaluation_samples.csv"
OUTPUT = PROJECT_ROOT / "data" / "manifests" / "kimi_prompt_development_analysis.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    attempts: list[dict] = []
    with RESPONSES.open(encoding="utf-8") as handle:
        for line in handle:
            attempts.append(json.loads(line))
    latest: dict[str, dict] = {}
    histories: dict[str, list[dict]] = {}
    for record in attempts:
        sample_id = record["sample_id"]
        latest[sample_id] = record
        histories.setdefault(sample_id, []).append(record)

    samples = pd.read_csv(SAMPLES)
    development = samples.loc[samples["cohort"] == "prompt_development"].copy()
    if len(development) != 80:
        raise ValueError(f"Expected 80 development samples, got {len(development)}")
    if set(latest) != set(development["sample_id"]):
        raise ValueError("Responses do not exactly cover the 80 development samples")
    if any(
        sample_id in set(samples.loc[samples["cohort"] != "prompt_development", "sample_id"])
        for sample_id in latest
    ):
        raise ValueError("Non-development sample detected in Kimi responses")

    rows = []
    for sample in development.itertuples(index=False):
        response = latest[sample.sample_id]
        parsed = response.get("parsed") or {}
        prediction = parsed.get("primary_evidence", "invalid")
        valid = response["status"] == "valid"
        correct = valid and prediction == sample.cross_seed_A_label
        rows.append(
            {
                "sample_id": sample.sample_id,
                "label": sample.cross_seed_A_label,
                "contrast": float(sample.contrast_median),
                "status": response["status"],
                "prediction": prediction,
                "text_share": parsed.get("claimed_text_share", np.nan),
                "confidence": parsed.get("confidence", np.nan),
                "correct": correct,
            }
        )
    frame = pd.DataFrame(rows)
    recalls = {}
    for label in ("text", "image"):
        group = frame.loc[frame["label"] == label]
        recalls[label] = float(group["correct"].mean())
    valid = frame.loc[frame["status"] == "valid"]
    spearman = float(valid[["text_share", "contrast"]].corr(method="spearman").iloc[0, 1])

    retry_samples = [values for values in histories.values() if len(values) > 1]
    retry_recovered = sum(values[-1]["status"] == "valid" for values in retry_samples)
    final_failures = [values[-1] for values in histories.values() if values[-1]["status"] != "valid"]
    usage_fields = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens")
    summary = {
        "status": "ANALYZED_DEVELOPMENT_ONLY",
        "scope": {
            "development_samples": len(frame),
            "confirmatory_samples": 0,
            "attempt_records": len(attempts),
        },
        "completion": {
            "final_status_counts": {
                str(key): int(value) for key, value in frame["status"].value_counts().items()
            },
            "retry_samples": len(retry_samples),
            "retry_recovered": retry_recovered,
            "retry_recovery_rate": float(retry_recovered / len(retry_samples)),
            "permanent_failures": len(final_failures),
            "permanent_failure_types": dict(Counter(r["error_type"] for r in final_failures)),
        },
        "predictions_valid_only": {
            str(key): int(value) for key, value in valid["prediction"].value_counts().items()
        },
        "intention_to_treat": {
            "accuracy": float(frame["correct"].mean()),
            "text_recall": recalls["text"],
            "image_recall": recalls["image"],
            "macro_recall": float(np.mean(list(recalls.values()))),
            "high_confidence_wrong_count": int(
                ((~frame["correct"]) & (frame["confidence"] >= 0.8)).sum()
            ),
            "high_confidence_wrong_rate": float(
                ((~frame["correct"]) & (frame["confidence"] >= 0.8)).mean()
            ),
        },
        "valid_only": {
            "accuracy": float(valid["correct"].mean()),
            "mean_confidence": float(valid["confidence"].mean()),
            "spearman_text_share_vs_contrast": spearman,
        },
        "usage_latest_success_or_failure": {
            field: sum(int(record.get("usage", {}).get(field, 0) or 0) for record in latest.values())
            for field in usage_fields
        },
        "usage_all_attempts": {
            field: sum(int(record.get("usage", {}).get(field, 0) or 0) for record in attempts)
            for field in usage_fields
        },
        "protocol_implication": (
            "Prompt v1 is not eligible for confirmatory use: 65/74 valid outputs "
            "selected both and no valid output selected image."
        ),
        "inputs": {
            "responses_sha256": sha256_file(RESPONSES),
            "samples_sha256": sha256_file(SAMPLES),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
