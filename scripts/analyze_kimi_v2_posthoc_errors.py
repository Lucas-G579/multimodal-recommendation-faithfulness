"""Run the frozen post-hoc diagnostic audit for Kimi v2."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "manifests" / "llm_evaluation_samples.csv"
DEV = ROOT / "outputs" / "llm_prompt_development_v2" / "kimi_responses.jsonl"
CONF = ROOT / "outputs" / "llm_confirmatory_v2" / "kimi_responses.jsonl"
OUTPUT = ROOT / "data" / "manifests" / "kimi_v2_posthoc_error_audit.json"
SEED = 20260811
ITERATIONS = 10_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_records(path: Path) -> tuple[list[dict], dict[str, dict], dict[str, list[dict]]]:
    attempts = [json.loads(line) for line in path.open(encoding="utf-8")]
    latest, histories = {}, {}
    for row in attempts:
        latest[row["sample_id"]] = row
        histories.setdefault(row["sample_id"], []).append(row)
    return attempts, latest, histories


def effective_status(row: dict) -> str:
    if row["status"] == "api_error" and row.get("error_type") == "ResponseValidationError":
        return "schema_invalid"
    return row["status"]


def make_frame(samples: pd.DataFrame, latest: dict[str, dict], cohort: str) -> pd.DataFrame:
    subset = samples.loc[samples["cohort"] == cohort]
    if set(subset["sample_id"]) != set(latest):
        raise ValueError(f"{cohort} response coverage mismatch")
    rows = []
    for sample in subset.itertuples(index=False):
        response = latest[sample.sample_id]
        parsed = response.get("parsed") or {}
        status = effective_status(response)
        prediction = parsed.get("primary_evidence", "invalid")
        rows.append({
            "sample_id": sample.sample_id,
            "user_id": int(sample.user_id),
            "bucket": int(sample.user_bucket),
            "label": sample.cross_seed_A_label,
            "contrast": float(sample.contrast_median),
            "abs_contrast": abs(float(sample.contrast_median)),
            "contrast_mad": float(sample.contrast_mad),
            "status": status,
            "prediction": prediction,
            "confidence": parsed.get("confidence", np.nan),
            "correct": bool(status == "valid" and prediction == sample.cross_seed_A_label),
        })
    return pd.DataFrame(rows)


def macro_recall(frame: pd.DataFrame) -> float:
    values = [frame.loc[frame["label"] == label, "correct"].mean() for label in ("text", "image")]
    return float(np.mean(values)) if all(np.isfinite(values)) else float("nan")


def metrics(frame: pd.DataFrame) -> dict:
    recalls = {
        label: float(frame.loc[frame["label"] == label, "correct"].mean())
        if (frame["label"] == label).any() else None
        for label in ("text", "image")
    }
    macro = macro_recall(frame)
    return {
        "samples": int(len(frame)),
        "unique_users": int(frame["user_id"].nunique()),
        "labels": dict(Counter(frame["label"])),
        "valid_rate": float((frame["status"] == "valid").mean()),
        "prediction_counts": dict(Counter(frame["prediction"])),
        "prediction_image_rate": float((frame["prediction"] == "image").mean()),
        "accuracy_itt": float(frame["correct"].mean()),
        "text_recall_itt": recalls["text"],
        "image_recall_itt": recalls["image"],
        "macro_recall_itt": float(macro) if np.isfinite(macro) else None,
    }


def distribution(frame: pd.DataFrame) -> dict:
    result = metrics(frame)
    result["buckets"] = sorted(int(x) for x in frame["bucket"].unique())
    for column in ("contrast", "abs_contrast", "contrast_mad"):
        result[column] = {
            "median": float(frame[column].median()),
            "q1": float(frame[column].quantile(0.25)),
            "q3": float(frame[column].quantile(0.75)),
        }
    return result


def bootstrap_delta(dev: pd.DataFrame, conf: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED)
    populations = []
    for frame in (dev, conf):
        users = np.array(sorted(frame["user_id"].unique()))
        groups = {u: frame.index[frame["user_id"] == u].to_numpy() for u in users}
        populations.append((frame, users, groups))
    accuracy, macro = [], []
    for _ in range(ITERATIONS):
        sampled = []
        for frame, users, groups in populations:
            chosen = rng.choice(users, size=len(users), replace=True)
            sampled.append(frame.loc[np.concatenate([groups[u] for u in chosen])])
        accuracy.append(float(sampled[1]["correct"].mean() - sampled[0]["correct"].mean()))
        delta = macro_recall(sampled[1]) - macro_recall(sampled[0])
        if np.isfinite(delta):
            macro.append(delta)
    return {
        "confirmatory_minus_development_accuracy": float(conf["correct"].mean() - dev["correct"].mean()),
        "accuracy_cluster_bootstrap_95ci": [float(x) for x in np.percentile(accuracy, [2.5, 97.5])],
        "confirmatory_minus_development_macro_recall": macro_recall(conf) - macro_recall(dev),
        "macro_recall_cluster_bootstrap_95ci": [float(x) for x in np.percentile(macro, [2.5, 97.5])],
    }


def main() -> None:
    samples = pd.read_csv(SAMPLES)
    dev_attempts, dev_latest, dev_histories = load_records(DEV)
    conf_attempts, conf_latest, conf_histories = load_records(CONF)
    dev = make_frame(samples, dev_latest, "prompt_development")
    conf = make_frame(samples, conf_latest, "primary_confirmatory")

    by_bucket = {str(bucket): metrics(group) for bucket, group in conf.groupby("bucket", sort=True)}

    strength = {}
    for label, group in conf.groupby("label", sort=True):
        group = group.copy()
        group["strength_quartile"] = pd.qcut(
            group["abs_contrast"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
        )
        strength[label] = {}
        for quartile, part in group.groupby("strength_quartile", observed=True, sort=True):
            strength[label][str(quartile)] = {
                **metrics(part),
                "abs_contrast_min": float(part["abs_contrast"].min()),
                "abs_contrast_max": float(part["abs_contrast"].max()),
                "mean_confidence_valid": float(part.loc[part["status"] == "valid", "confidence"].mean()),
            }

    permanent = conf.loc[conf["status"] != "valid"]
    failures = {
        "first_attempt_nonvalid": sum(effective_status(h[0]) != "valid" for h in conf_histories.values()),
        "retry_samples": sum(len(h) > 1 for h in conf_histories.values()),
        "retry_recovered": sum(len(h) > 1 and effective_status(h[-1]) == "valid" for h in conf_histories.values()),
        "permanent_statuses": dict(Counter(permanent["status"])),
        "permanent_by_label": dict(Counter(permanent["label"])),
        "permanent_by_bucket": {str(k): int(v) for k, v in Counter(permanent["bucket"]).items()},
    }

    valid = conf.loc[conf["status"] == "valid"].copy()
    valid["confidence_bin"] = pd.cut(
        valid["confidence"], bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0000001],
        right=False, include_lowest=True,
    )
    confidence = {}
    for interval in valid["confidence_bin"].cat.categories:
        group = valid.loc[valid["confidence_bin"] == interval]
        confidence[str(interval)] = {
            "samples": int(len(group)),
            "accuracy": float(group["correct"].mean()) if len(group) else None,
            "labels": dict(Counter(group["label"])),
        }
    confidence["high_confidence_wrong"] = int(((~valid["correct"]) & (valid["confidence"] >= 0.8)).sum())

    summary = {
        "status": "POSTHOC_EXPLORATORY_COMPLETE",
        "protocol_commit": "2ae04bd",
        "development_vs_confirmatory": {
            "development": distribution(dev),
            "confirmatory": distribution(conf),
        },
        "confirmatory_by_user_bucket": by_bucket,
        "confirmatory_by_label_and_strength_quartile": strength,
        "failure_mechanism": failures,
        "confidence_diagnostic": confidence,
        "development_to_confirmatory_delta": bootstrap_delta(dev, conf),
        "bootstrap": {"unit": "user_id", "iterations": ITERATIONS, "seed": SEED},
        "attempt_counts": {"development": len(dev_attempts), "confirmatory": len(conf_attempts)},
        "inputs": {
            "samples_sha256": sha256(SAMPLES),
            "development_responses_sha256": sha256(DEV),
            "confirmatory_responses_sha256": sha256(CONF),
        },
    }
    serialized = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    OUTPUT.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
