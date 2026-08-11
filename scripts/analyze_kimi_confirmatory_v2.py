"""Analyze the frozen one-shot Kimi v2 confirmatory evaluation."""

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
RESPONSES = PROJECT_ROOT / "outputs" / "llm_confirmatory_v2" / "kimi_responses.jsonl"
OUTPUT = PROJECT_ROOT / "data" / "manifests" / "kimi_confirmatory_v2_analysis.json"
BOOTSTRAP_SEED = 20260811
BOOTSTRAP_ITERATIONS = 10_000


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def interval(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return [float(x) for x in np.percentile(array, [2.5, 97.5])]


def macro_recall(frame: pd.DataFrame) -> float:
    return float(np.mean([
        frame.loc[frame["label"] == label, "correct"].mean()
        for label in ("text", "image")
    ]))


def main() -> None:
    samples = pd.read_csv(SAMPLES)
    samples = samples.loc[samples["cohort"] == "primary_confirmatory"].copy()
    if len(samples) != 590 or samples["sample_id"].nunique() != 590:
        raise ValueError("Expected exactly 590 unique confirmatory samples")

    attempts = [json.loads(line) for line in RESPONSES.open(encoding="utf-8")]
    latest: dict[str, dict] = {}
    histories: dict[str, list[dict]] = {}
    for record in attempts:
        latest[record["sample_id"]] = record
        histories.setdefault(record["sample_id"], []).append(record)
    expected = set(samples["sample_id"])
    if set(latest) != expected:
        raise ValueError("Responses do not exactly cover the confirmatory sample IDs")

    rows = []
    for sample in samples.itertuples(index=False):
        record = latest[sample.sample_id]
        parsed = record.get("parsed") or {}
        valid = record["status"] == "valid"
        prediction = parsed.get("primary_evidence", "invalid")
        rows.append({
            "sample_id": sample.sample_id,
            "user_id": int(sample.user_id),
            "label": sample.cross_seed_A_label,
            "contrast": float(sample.contrast_median),
            "status": record["status"],
            "prediction": prediction,
            "text_share": parsed.get("claimed_text_share", np.nan),
            "confidence": parsed.get("confidence", np.nan),
            "correct": bool(valid and prediction == sample.cross_seed_A_label),
        })
    frame = pd.DataFrame(rows)
    valid = frame.loc[frame["status"] == "valid"].copy()

    users = np.array(sorted(frame["user_id"].unique()))
    groups = {user: frame.index[frame["user_id"] == user].to_numpy() for user in users}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    accuracy_boot, macro_boot, spearman_boot = [], [], []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sampled_users = rng.choice(users, size=len(users), replace=True)
        sampled = frame.loc[np.concatenate([groups[user] for user in sampled_users])]
        accuracy_boot.append(float(sampled["correct"].mean()))
        macro_boot.append(macro_recall(sampled))
        sampled_valid = sampled.loc[sampled["status"] == "valid"]
        rho = spearmanr(sampled_valid["text_share"], sampled_valid["contrast"]).statistic
        if np.isfinite(rho):
            spearman_boot.append(float(rho))

    recalls = {
        label: float(frame.loc[frame["label"] == label, "correct"].mean())
        for label in ("text", "image")
    }
    valid_recalls = {
        label: float(valid.loc[valid["label"] == label, "correct"].mean())
        for label in ("text", "image")
    }
    retry_histories = [history for history in histories.values() if len(history) > 1]
    permanent = [history[-1] for history in histories.values() if history[-1]["status"] != "valid"]
    usage_fields = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens")
    confusion = pd.crosstab(frame["label"], frame["prediction"], dropna=False)

    summary = {
        "status": "CONFIRMATORY_ANALYZED_FROZEN",
        "scope": {"confirmatory_samples": 590, "development_samples": 0},
        "class_counts": dict(Counter(frame["label"])),
        "execution": {
            "attempt_records": len(attempts),
            "final_status_counts": dict(Counter(frame["status"])),
            "retry_samples": len(retry_histories),
            "retry_recovered": sum(h[-1]["status"] == "valid" for h in retry_histories),
            "permanent_failure_types": dict(Counter(r["status"] for r in permanent)),
            "usage_all_attempts": {
                field: sum(int(r.get("usage", {}).get(field, 0) or 0) for r in attempts)
                for field in usage_fields
            },
        },
        "predictions_valid_only": dict(Counter(valid["prediction"])),
        "confusion_intention_to_treat": {
            str(label): {str(pred): int(value) for pred, value in row.items()}
            for label, row in confusion.to_dict(orient="index").items()
        },
        "intention_to_treat_primary": {
            "majority_text_baseline_accuracy": float((frame["label"] == "text").mean()),
            "majority_text_baseline_macro_recall": 0.5,
            "accuracy": float(frame["correct"].mean()),
            "accuracy_cluster_bootstrap_95ci": interval(accuracy_boot),
            "text_recall": recalls["text"],
            "image_recall": recalls["image"],
            "macro_recall": macro_recall(frame),
            "macro_recall_cluster_bootstrap_95ci": interval(macro_boot),
            "high_confidence_wrong_count": int(
                ((~frame["correct"]) & (frame["confidence"] >= 0.8)).sum()
            ),
            "image_prediction_precision": float(
                ((frame["prediction"] == "image") & (frame["label"] == "image")).sum()
                / (frame["prediction"] == "image").sum()
            ),
        },
        "valid_only_secondary": {
            "samples": int(len(valid)),
            "accuracy": float(valid["correct"].mean()),
            "text_recall": valid_recalls["text"],
            "image_recall": valid_recalls["image"],
            "macro_recall": float(np.mean(list(valid_recalls.values()))),
            "mean_confidence": float(valid["confidence"].mean()),
            "spearman_text_share_vs_contrast": float(
                spearmanr(valid["text_share"], valid["contrast"]).statistic
            ),
            "spearman_cluster_bootstrap_95ci": interval(spearman_boot),
        },
        "bootstrap": {
            "unit": "user_id",
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "unique_users": int(len(users)),
        },
        "protocol": {
            "version": "v2-forced-choice",
            "model": "kimi-k2.6",
            "thinking": "disabled",
            "max_completion_tokens": 512,
            "max_attempts_per_sample": 2,
            "system_prompt_sha256": "174BAE1BC178B1A1B253377848898688439667EE02CE41FB6ECD937785900FB4",
            "user_template_sha256": "D6C62EED6077CDFBD147532ADAF2545D4666ACA59894385DC5007355C044B4C7",
            "response_schema_sha256": "A1DF34AADBBECD507CCF91D821FC321D7A08B3634EBC22FEF42C8DE51BDFCA04",
        },
        "inputs": {
            "samples_sha256": sha256_file(SAMPLES),
            "responses_sha256": sha256_file(RESPONSES),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
