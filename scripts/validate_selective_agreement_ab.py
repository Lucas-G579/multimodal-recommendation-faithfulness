"""Validate the frozen Mean/Max agreement rule on untouched A+B samples."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_non_llm_modality_baselines import (
    BLIND,
    IMAGE,
    SAMPLES,
    TEXT,
    cosine_catalog,
    item_id_from_path,
    normalize,
    percentile,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "manifests" / "selective_agreement_ab_validation.json"
PROTOCOL_COMMIT = "9d1a274"
SEED = 20260811
ITERATIONS = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def metrics(frame: pd.DataFrame, prediction: str) -> dict:
    correct = frame[prediction] == frame["label"]
    recalls = {
        label: float(correct.loc[frame["label"] == label].mean())
        if (frame["label"] == label).any() else None
        for label in ("text", "image")
    }
    finite = [value for value in recalls.values() if value is not None]
    table = pd.crosstab(frame["label"], frame[prediction])
    return {
        "samples": int(len(frame)),
        "accuracy": float(correct.mean()) if len(frame) else None,
        "text_recall": recalls["text"],
        "image_recall": recalls["image"],
        "macro_recall": float(np.mean(finite)) if len(finite) == 2 else None,
        "prediction_counts": dict(Counter(frame[prediction])),
        "confusion": {
            str(label): {str(pred): int(value) for pred, value in row.items()}
            for label, row in table.to_dict(orient="index").items()
        },
    }


def fast_macro(frame: pd.DataFrame, prediction: str) -> float:
    correct = (frame[prediction] == frame["label"]).to_numpy()
    labels = frame["label"].to_numpy()
    values = [correct[labels == label].mean() for label in ("text", "image")]
    return float(np.mean(values)) if all(np.isfinite(values)) else float("nan")


def bootstrap(frame: pd.DataFrame) -> dict:
    users = np.array(sorted(frame["user_id"].unique()))
    groups = {user: frame.index[frame["user_id"] == user].to_numpy() for user in users}
    rng = np.random.default_rng(SEED)
    values = {key: [] for key in (
        "coverage", "selective_answered_macro", "selective_end_to_end_macro",
        "mean_macro", "max_macro", "selective_minus_mean_macro",
    )}
    for _ in range(ITERATIONS):
        chosen = rng.choice(users, size=len(users), replace=True)
        sample = frame.loc[np.concatenate([groups[user] for user in chosen])]
        answered = sample.loc[sample["selective_prediction"] != "abstain"]
        selective_macro = fast_macro(answered, "selective_prediction")
        mean_macro = fast_macro(sample, "mean_prediction")
        values["coverage"].append(float(len(answered) / len(sample)))
        values["selective_answered_macro"].append(selective_macro)
        values["selective_end_to_end_macro"].append(fast_macro(sample, "selective_prediction"))
        values["mean_macro"].append(mean_macro)
        values["max_macro"].append(fast_macro(sample, "max_prediction"))
        values["selective_minus_mean_macro"].append(selective_macro - mean_macro)
    result = {}
    for key, series in values.items():
        finite = np.asarray(series, dtype=float)
        finite = finite[np.isfinite(finite)]
        result[key] = {
            "valid_iterations": int(len(finite)),
            "95ci": [float(x) for x in np.percentile(finite, [2.5, 97.5])],
        }
    return result


def main() -> None:
    all_samples = pd.read_csv(SAMPLES)
    samples = all_samples.loc[all_samples["cohort"] == "A_or_B_sensitivity"].copy()
    if len(samples) != 400 or samples["user_id"].nunique() != 400:
        raise ValueError("Expected 400 user-isolated A+B samples")
    if Counter(samples["cross_seed_A_or_B_label"]) != Counter({"text": 200, "image": 200}):
        raise ValueError("Expected balanced A+B labels")
    target_ids = set(samples["sample_id"])
    blind = {}
    for line in BLIND.open(encoding="utf-8"):
        row = json.loads(line)
        if row["sample_id"] in target_ids:
            blind[row["sample_id"]] = row
    if set(blind) != target_ids:
        raise ValueError("Blind input coverage mismatch")

    image, image_zero = normalize(np.load(IMAGE, mmap_mode="r"))
    text, text_zero = normalize(np.load(TEXT, mmap_mode="r"))
    targets = sorted({int(item) for item in samples["item_id"]})
    target_row = {item: index for index, item in enumerate(targets)}
    image_catalog, image_backend = cosine_catalog(targets, image)
    text_catalog, text_backend = cosine_catalog(targets, text)
    rows, missing_image, missing_text = [], 0, 0
    for sample in samples.itertuples(index=False):
        request = blind[sample.sample_id]
        target_id = int(sample.item_id)
        index = target_row[target_id]
        image_values, text_values = [], []
        for history in request["history"]:
            history_id = item_id_from_path(history.get("image_path", ""))
            if history_id is not None and history.get("image_available", False):
                image_values.append(percentile(image_catalog[index], target_id, history_id))
            else:
                missing_image += 1
            if history_id is not None and history.get("title", ""):
                text_values.append(percentile(text_catalog[index], target_id, history_id))
            else:
                missing_text += 1
        if not image_values or not text_values:
            raise ValueError(f"No usable history for {sample.sample_id}")
        image_mean, text_mean = float(np.mean(image_values)), float(np.mean(text_values))
        image_max, text_max = float(np.max(image_values)), float(np.max(text_values))
        mean_prediction = "image" if image_mean > text_mean else "text"
        max_prediction = "image" if image_max > text_max else "text"
        rows.append({
            "sample_id": sample.sample_id,
            "user_id": int(sample.user_id),
            "label": sample.cross_seed_A_or_B_label,
            "mean_prediction": mean_prediction,
            "max_prediction": max_prediction,
            "selective_prediction": mean_prediction if mean_prediction == max_prediction else "abstain",
            "majority_text": "text",
        })
    frame = pd.DataFrame(rows)
    answered = frame.loc[frame["selective_prediction"] != "abstain"]
    coverage = float(len(answered) / len(frame))
    coverage_by_label = {
        label: float((group["selective_prediction"] != "abstain").mean())
        for label, group in frame.groupby("label", sort=True)
    }
    results = {
        "full_mean": metrics(frame, "mean_prediction"),
        "full_max": metrics(frame, "max_prediction"),
        "majority_text": metrics(frame, "majority_text"),
        "selective_answered": metrics(answered, "selective_prediction"),
        "selective_end_to_end_abstain_wrong": metrics(frame, "selective_prediction"),
        "agreement_groups": {
            "agree": {
                "samples": int(len(answered)),
                "mean_rule": metrics(answered, "mean_prediction"),
                "max_rule": metrics(answered, "max_prediction"),
            },
            "disagree": {
                "samples": int(len(frame) - len(answered)),
                "mean_rule": metrics(frame.loc[frame["selective_prediction"] == "abstain"], "mean_prediction"),
                "max_rule": metrics(frame.loc[frame["selective_prediction"] == "abstain"], "max_prediction"),
            },
        },
    }
    uncertainty = bootstrap(frame)
    primary_ci = uncertainty["selective_answered_macro"]["95ci"]
    success = coverage >= 0.5 and primary_ci[0] > 0.5
    summary = {
        "status": "INDEPENDENT_SENSITIVITY_VALIDATION_COMPLETE",
        "protocol_commit": PROTOCOL_COMMIT,
        "cohort": {
            "name": "A_or_B_sensitivity",
            "samples": 400,
            "unique_users": 400,
            "labels": {"text": 200, "image": 200},
            "user_overlap_with_other_cohorts": 0,
            "strict_A_equivalent": False,
        },
        "primary_endpoint": {
            "coverage": coverage,
            "coverage_by_label": coverage_by_label,
            "selective_answered_macro_recall": results["selective_answered"]["macro_recall"],
            "selective_answered_macro_recall_95ci": primary_ci,
            "success_gate_coverage_at_least_0_5": coverage >= 0.5,
            "success_gate_ci_lower_above_0_5": primary_ci[0] > 0.5,
            "overall_success": success,
        },
        "results": results,
        "cluster_bootstrap": uncertainty,
        "bootstrap": {"unit": "user_id", "iterations": ITERATIONS, "seed": SEED},
        "integrity": {
            "evaluated_sample_ids": len(set(frame["sample_id"])),
            "unstable_overconfidence_predictions_computed": 0,
            "unique_targets": len(targets),
            "missing_image_history_entries": missing_image,
            "missing_text_history_entries": missing_text,
            "zero_norm_vectors": {"image": image_zero, "text": text_zero},
            "compute_backend": {"image": image_backend, "text": text_backend},
        },
        "inputs": {
            "samples_sha256": sha256(SAMPLES),
            "blind_sha256": sha256(BLIND),
            "image_sha256": sha256(IMAGE),
            "text_sha256": sha256(TEXT),
        },
    }
    serialized = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    OUTPUT.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
