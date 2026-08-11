"""Evaluate frozen deterministic non-LLM modality attribution baselines."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "manifests" / "llm_evaluation_samples.csv"
BLIND = ROOT / "data" / "manifests" / "llm_blind_inputs.jsonl"
IMAGE = ROOT / "external" / "MMRec" / "data" / "baby" / "image_feat.npy"
TEXT = ROOT / "external" / "MMRec" / "data" / "baby" / "text_feat.npy"
KIMI_DEV = ROOT / "outputs" / "llm_prompt_development_v2" / "kimi_responses.jsonl"
KIMI_CONF = ROOT / "outputs" / "llm_confirmatory_v2" / "kimi_responses.jsonl"
OUTPUT = ROOT / "data" / "manifests" / "non_llm_modality_baselines.json"
SEED = 20260811
ITERATIONS = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def item_id_from_path(value: str) -> int | None:
    match = re.search(r"/(\d+)\.img$", value.replace("\\", "/"))
    return int(match.group(1)) if match else None


def normalize(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    matrix = np.asarray(matrix, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains non-finite values")
    norms = np.linalg.norm(matrix, axis=1)
    zero = int((norms == 0).sum())
    safe = np.where(norms == 0, 1.0, norms)
    return matrix / safe[:, None], zero


def cosine_catalog(target_ids: list[int], features: np.ndarray) -> tuple[np.ndarray, str]:
    targets = features[np.asarray(target_ids)]
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.device("cuda")
            with torch.inference_mode():
                result = (
                    torch.from_numpy(targets).to(device)
                    @ torch.from_numpy(features).to(device).T
                ).cpu().numpy()
            return result, "cuda"
    except (ImportError, RuntimeError):
        pass
    return targets @ features.T, "cpu"


def percentile(similarities: np.ndarray, target_id: int, history_id: int) -> float:
    value = similarities[history_id]
    less_or_equal = int(np.count_nonzero(similarities <= value))
    if similarities[target_id] <= value:
        less_or_equal -= 1
    return less_or_equal / (len(similarities) - 1)


def load_kimi(path: Path) -> dict[str, dict]:
    latest = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        latest[row["sample_id"]] = row
    return latest


def score(frame: pd.DataFrame, prediction: str) -> dict:
    correct = frame[prediction] == frame["label"]
    recalls = {
        label: float(correct.loc[frame["label"] == label].mean())
        for label in ("text", "image")
    }
    table = pd.crosstab(frame["label"], frame[prediction])
    return {
        "accuracy": float(correct.mean()),
        "text_recall": recalls["text"],
        "image_recall": recalls["image"],
        "macro_recall": float(np.mean(list(recalls.values()))),
        "prediction_image_rate": float((frame[prediction] == "image").mean()),
        "confusion": {
            str(label): {str(pred): int(value) for pred, value in row.items()}
            for label, row in table.to_dict(orient="index").items()
        },
    }


def bootstrap(frame: pd.DataFrame, methods: list[str]) -> dict:
    users = np.array(sorted(frame["user_id"].unique()))
    groups = {u: frame.index[frame["user_id"] == u].to_numpy() for u in users}
    rng = np.random.default_rng(SEED)
    values = {m: {"accuracy": [], "macro": []} for m in methods}
    deltas = {"mean_minus_kimi_accuracy": [], "mean_minus_kimi_macro": [], "mean_minus_majority_macro": []}

    def fast_metrics(sample: pd.DataFrame, method: str) -> tuple[float, float]:
        correct = (sample[method] == sample["label"]).to_numpy()
        labels = sample["label"].to_numpy()
        accuracy = float(correct.mean())
        macro = float(np.mean([correct[labels == label].mean() for label in ("text", "image")]))
        return accuracy, macro

    for _ in range(ITERATIONS):
        chosen = rng.choice(users, size=len(users), replace=True)
        sample = frame.loc[np.concatenate([groups[u] for u in chosen])]
        for method in methods:
            accuracy, macro = fast_metrics(sample, method)
            values[method]["accuracy"].append(accuracy)
            values[method]["macro"].append(macro)
        mean_accuracy, mean_macro = fast_metrics(sample, "mean_percentile")
        kimi_accuracy, kimi_macro = fast_metrics(sample, "kimi")
        _, majority_macro = fast_metrics(sample, "majority_text")
        deltas["mean_minus_kimi_accuracy"].append(mean_accuracy - kimi_accuracy)
        deltas["mean_minus_kimi_macro"].append(mean_macro - kimi_macro)
        deltas["mean_minus_majority_macro"].append(mean_macro - majority_macro)
    result = {}
    for method in methods:
        result[method] = {
            "accuracy_95ci": [float(x) for x in np.percentile(values[method]["accuracy"], [2.5, 97.5])],
            "macro_recall_95ci": [float(x) for x in np.percentile(values[method]["macro"], [2.5, 97.5])],
        }
    observed = {
        "mean_minus_kimi_accuracy": score(frame, "mean_percentile")["accuracy"] - score(frame, "kimi")["accuracy"],
        "mean_minus_kimi_macro": score(frame, "mean_percentile")["macro_recall"] - score(frame, "kimi")["macro_recall"],
        "mean_minus_majority_macro": score(frame, "mean_percentile")["macro_recall"] - score(frame, "majority_text")["macro_recall"],
    }
    result["paired_deltas"] = {
        name: {
            "estimate": float(observed[name]),
            "95ci": [float(x) for x in np.percentile(series, [2.5, 97.5])],
        }
        for name, series in deltas.items()
    }
    return result


def main() -> None:
    samples = pd.read_csv(SAMPLES)
    samples = samples.loc[samples["cohort"].isin(["prompt_development", "primary_confirmatory"])].copy()
    blind = {row["sample_id"]: row for row in map(json.loads, BLIND.open(encoding="utf-8"))}
    if set(samples["sample_id"]) != set(blind).intersection(set(samples["sample_id"])):
        raise ValueError("Blind inputs do not cover all evaluated samples")

    image, image_zero = normalize(np.load(IMAGE, mmap_mode="r"))
    text, text_zero = normalize(np.load(TEXT, mmap_mode="r"))
    if image.shape[0] != 7050 or text.shape[0] != 7050:
        raise ValueError("Expected 7050 catalog items")

    target_ids = [int(sample_id.rsplit("-i", 1)[1]) for sample_id in samples["sample_id"]]
    unique_targets = sorted(set(target_ids))
    target_row = {item: idx for idx, item in enumerate(unique_targets)}
    image_catalog, image_backend = cosine_catalog(unique_targets, image)
    text_catalog, text_backend = cosine_catalog(unique_targets, text)
    kimi_by_cohort = {
        "prompt_development": load_kimi(KIMI_DEV),
        "primary_confirmatory": load_kimi(KIMI_CONF),
    }

    rows, missing_image_history, missing_text_history = [], 0, 0
    mean_ties = max_ties = 0
    for sample in samples.itertuples(index=False):
        request = blind[sample.sample_id]
        target_id = int(sample.item_id)
        if target_id != int(sample.sample_id.rsplit("-i", 1)[1]):
            raise ValueError("Target item ID mismatch")
        row_index = target_row[target_id]
        image_percentiles, text_percentiles = [], []
        for history in request["history"]:
            history_id = item_id_from_path(history.get("image_path", ""))
            if history_id is None or not history.get("image_available", False):
                missing_image_history += 1
            else:
                image_percentiles.append(percentile(image_catalog[row_index], target_id, history_id))
            if history_id is None or not history.get("title", ""):
                missing_text_history += 1
            else:
                text_percentiles.append(percentile(text_catalog[row_index], target_id, history_id))
        if not image_percentiles or not text_percentiles:
            raise ValueError(f"No usable history for {sample.sample_id}")
        image_mean, text_mean = float(np.mean(image_percentiles)), float(np.mean(text_percentiles))
        image_max, text_max = float(np.max(image_percentiles)), float(np.max(text_percentiles))
        mean_ties += image_mean == text_mean
        max_ties += image_max == text_max
        kimi_row = kimi_by_cohort[sample.cohort][sample.sample_id]
        parsed = kimi_row.get("parsed") or {}
        kimi = parsed.get("primary_evidence", "invalid") if kimi_row["status"] == "valid" else "invalid"
        rows.append({
            "sample_id": sample.sample_id,
            "user_id": int(sample.user_id),
            "cohort": sample.cohort,
            "label": sample.cross_seed_A_label,
            "mean_percentile": "image" if image_mean > text_mean else "text",
            "max_percentile": "image" if image_max > text_max else "text",
            "majority_text": "text",
            "seed_999_reference": sample.seed_999_strict_label,
            "label_identity": sample.cross_seed_A_label,
            "kimi": kimi,
        })
    frame = pd.DataFrame(rows)
    methods = ["mean_percentile", "max_percentile", "majority_text", "seed_999_reference", "label_identity", "kimi"]
    cohorts = {}
    for cohort, group in frame.groupby("cohort", sort=True):
        cohorts[cohort] = {method: score(group, method) for method in methods}
    confirmatory = frame.loc[frame["cohort"] == "primary_confirmatory"]
    summary = {
        "status": "POSTHOC_EXPLORATORY_COMPLETE",
        "protocol_commit": "eefcfca",
        "primary_baseline": "mean_percentile",
        "cohorts": cohorts,
        "confirmatory_cluster_bootstrap": bootstrap(confirmatory, methods),
        "bootstrap": {"unit": "user_id", "iterations": ITERATIONS, "seed": SEED},
        "integrity": {
            "evaluated_samples": int(len(frame)),
            "unique_targets": len(unique_targets),
            "feature_shapes": {"image": list(image.shape), "text": list(text.shape)},
            "zero_norm_vectors": {"image": image_zero, "text": text_zero},
            "missing_image_history_entries": missing_image_history,
            "missing_text_history_entries": missing_text_history,
            "mean_percentile_ties": int(mean_ties),
            "max_percentile_ties": int(max_ties),
            "compute_backend": {"image": image_backend, "text": text_backend},
        },
        "inputs": {
            "samples_sha256": sha256(SAMPLES),
            "blind_inputs_sha256": sha256(BLIND),
            "image_features_sha256": sha256(IMAGE),
            "text_features_sha256": sha256(TEXT),
            "kimi_development_sha256": sha256(KIMI_DEV),
            "kimi_confirmatory_sha256": sha256(KIMI_CONF),
        },
    }
    serialized = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    OUTPUT.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
