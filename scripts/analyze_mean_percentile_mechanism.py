"""Run the frozen Day 18 mechanism audit and generate paper-ready figures."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_non_llm_modality_baselines import (
    BLIND,
    IMAGE,
    KIMI_CONF,
    SAMPLES,
    TEXT,
    cosine_catalog,
    item_id_from_path,
    load_kimi,
    normalize,
    percentile,
)


ROOT = Path(__file__).resolve().parents[1]
DAY17 = ROOT / "data" / "manifests" / "non_llm_modality_baselines.json"
OUTPUT = ROOT / "data" / "manifests" / "mean_percentile_mechanism.json"
FIGURES = ROOT / "results" / "figures"
PROTOCOL_COMMIT = "29f8018"
MARGIN_EDGES = [0.0, 0.025, 0.05, 0.10, 0.20, 1.0000001]
MARGIN_LABELS = ["[0,.025)", "[.025,.05)", "[.05,.10)", "[.10,.20)", "[.20,1]"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def metrics(frame: pd.DataFrame, prediction: str) -> dict:
    correct = frame[prediction] == frame["label"]
    recalls = {}
    for label in ("text", "image"):
        mask = frame["label"] == label
        recalls[label] = float(correct.loc[mask].mean()) if mask.any() else None
    finite = [x for x in recalls.values() if x is not None]
    return {
        "samples": int(len(frame)),
        "accuracy": float(correct.mean()) if len(frame) else None,
        "text_samples": int((frame["label"] == "text").sum()),
        "image_samples": int((frame["label"] == "image").sum()),
        "text_recall": recalls["text"],
        "image_recall": recalls["image"],
        "macro_recall": float(np.mean(finite)) if len(finite) == 2 else None,
    }


def save_figure(fig: plt.Figure, stem: str) -> list[str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "svg"):
        path = FIGURES / f"{stem}.{suffix}"
        metadata = {"Software": "FaithRec-MM"} if suffix == "png" else {"Date": None}
        fig.savefig(path, dpi=300, bbox_inches="tight", metadata=metadata)
        outputs.append(str(path.relative_to(ROOT)).replace("\\", "/"))
    plt.close(fig)
    return outputs


def build_frame() -> tuple[pd.DataFrame, dict]:
    samples = pd.read_csv(SAMPLES)
    samples = samples.loc[samples["cohort"] == "primary_confirmatory"].copy()
    blind = {row["sample_id"]: row for row in map(json.loads, BLIND.open(encoding="utf-8"))}
    image, image_zero = normalize(np.load(IMAGE, mmap_mode="r"))
    text, text_zero = normalize(np.load(TEXT, mmap_mode="r"))
    targets = sorted({int(x) for x in samples["item_id"]})
    target_row = {item: index for index, item in enumerate(targets)}
    image_catalog, image_backend = cosine_catalog(targets, image)
    text_catalog, text_backend = cosine_catalog(targets, text)
    kimi = load_kimi(KIMI_CONF)
    rows = []
    for sample in samples.itertuples(index=False):
        request = blind[sample.sample_id]
        target_id = int(sample.item_id)
        index = target_row[target_id]
        image_values, text_values = [], []
        for history in request["history"]:
            history_id = item_id_from_path(history.get("image_path", ""))
            if history_id is not None and history.get("image_available", False):
                image_values.append(percentile(image_catalog[index], target_id, history_id))
            if history_id is not None and history.get("title", ""):
                text_values.append(percentile(text_catalog[index], target_id, history_id))
        image_mean, text_mean = float(np.mean(image_values)), float(np.mean(text_values))
        image_max, text_max = float(np.max(image_values)), float(np.max(text_values))
        response = kimi[sample.sample_id]
        parsed = response.get("parsed") or {}
        kimi_prediction = parsed.get("primary_evidence", "invalid") if response["status"] == "valid" else "invalid"
        rows.append({
            "sample_id": sample.sample_id,
            "user_id": int(sample.user_id),
            "target_title": request["target"].get("title", ""),
            "label": sample.cross_seed_A_label,
            "image_mean_percentile": image_mean,
            "text_mean_percentile": text_mean,
            "image_max_percentile": image_max,
            "text_max_percentile": text_max,
            "margin": image_mean - text_mean,
            "abs_margin": abs(image_mean - text_mean),
            "mean_prediction": "image" if image_mean > text_mean else "text",
            "max_prediction": "image" if image_max > text_max else "text",
            "kimi_prediction": kimi_prediction,
        })
    frame = pd.DataFrame(rows)
    frame["mean_correct"] = frame["mean_prediction"] == frame["label"]
    frame["max_correct"] = frame["max_prediction"] == frame["label"]
    frame["kimi_correct"] = frame["kimi_prediction"] == frame["label"]
    integrity = {
        "samples": int(len(frame)),
        "unique_users": int(frame["user_id"].nunique()),
        "unique_targets": len(targets),
        "zero_norm_vectors": {"image": image_zero, "text": text_zero},
        "compute_backend": {"image": image_backend, "text": text_backend},
    }
    return frame, integrity


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.hashsalt": "faithrec-day18-v1",
    })
    frame, integrity = build_frame()
    day17 = json.loads(DAY17.read_text(encoding="utf-8"))
    expected = day17["cohorts"]["primary_confirmatory"]["mean_percentile"]
    observed = metrics(frame, "mean_prediction")
    for key in ("accuracy", "text_recall", "image_recall", "macro_recall"):
        if not np.isclose(observed[key], expected[key], atol=1e-12):
            raise ValueError(f"Day 17 metric mismatch: {key}")

    frame["margin_bin"] = pd.cut(
        frame["abs_margin"], bins=MARGIN_EDGES, labels=MARGIN_LABELS,
        right=False, include_lowest=True,
    )
    margin_bins = {}
    for label in MARGIN_LABELS:
        group = frame.loc[frame["margin_bin"] == label]
        margin_bins[label] = metrics(group, "mean_prediction")

    frame["mean_max_group"] = np.where(
        frame["mean_prediction"] == frame["max_prediction"], "agree", "disagree"
    )
    mean_max = {
        name: {
            "coverage": float(len(group) / len(frame)),
            "mean_rule": metrics(group, "mean_prediction"),
            "max_rule": metrics(group, "max_prediction"),
        }
        for name, group in frame.groupby("mean_max_group", sort=True)
    }

    frame["comparison_group"] = np.select(
        [
            frame["mean_correct"] & frame["kimi_correct"],
            frame["mean_correct"] & ~frame["kimi_correct"],
            ~frame["mean_correct"] & frame["kimi_correct"],
        ],
        ["both_correct", "mean_only", "kimi_only"],
        default="neither",
    )
    comparison = {
        "overall": dict(Counter(frame["comparison_group"])),
        "by_label": {
            label: dict(Counter(group["comparison_group"]))
            for label, group in frame.groupby("label", sort=True)
        },
    }

    cases = {}
    for label in ("text", "image"):
        wrong = frame.loc[(frame["label"] == label) & ~frame["mean_correct"]].sort_values(
            ["abs_margin", "sample_id"], ascending=[False, True]
        ).head(5)
        correct = frame.loc[(frame["label"] == label) & frame["mean_correct"]].sort_values(
            ["abs_margin", "sample_id"], ascending=[True, True]
        ).head(5)
        columns = [
            "sample_id", "target_title", "label", "mean_prediction", "kimi_prediction",
            "image_mean_percentile", "text_mean_percentile", "margin", "abs_margin",
        ]
        cases[label] = {
            "strongest_margin_errors": wrong[columns].to_dict(orient="records"),
            "smallest_margin_correct": correct[columns].to_dict(orient="records"),
        }

    methods = ["kimi", "majority_text", "mean_percentile", "max_percentile"]
    names = ["Kimi v2", "Majority text", "Mean percentile", "Max percentile"]
    colors = ["#CC6677", "#999999", "#4477AA", "#228833"]
    cohort_results = day17["cohorts"]["primary_confirmatory"]
    bootstrap = day17["confirmatory_cluster_bootstrap"]

    values = [cohort_results[m]["macro_recall"] for m in methods]
    intervals = [bootstrap[m]["macro_recall_95ci"] for m in methods]
    errors = np.array([[v - ci[0] for v, ci in zip(values, intervals)], [ci[1] - v for v, ci in zip(values, intervals)]])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(names, values, color=colors, yerr=errors, capsize=4)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Balanced-chance reference")
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Macro recall")
    ax.set_title("Confirmatory-set modality attribution\n(Post-hoc exploratory comparison)")
    ax.legend(frameon=False, loc="upper left")
    ax.tick_params(axis="x", rotation=15)
    figure_files = save_figure(fig, "day18_macro_recall_comparison")

    x = np.arange(len(names))
    width = 0.36
    text_recalls = [cohort_results[m]["text_recall"] for m in methods]
    image_recalls = [cohort_results[m]["image_recall"] for m in methods]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(x - width / 2, text_recalls, width, label="Text recall", color="#4477AA")
    ax.bar(x + width / 2, image_recalls, width, label="Image recall", color="#EE6677")
    ax.set_xticks(x, names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall")
    ax.set_title("Class-specific recall reveals majority-class masking\n(Post-hoc exploratory)")
    ax.legend(frameon=False)
    figure_files += save_figure(fig, "day18_class_recall_comparison")

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for label, color in (("text", "#4477AA"), ("image", "#EE6677")):
        for correct, marker, alpha in ((True, "o", 0.55), (False, "x", 0.7)):
            subset = frame.loc[(frame["label"] == label) & (frame["mean_correct"] == correct)]
            ax.scatter(
                subset["text_mean_percentile"], subset["image_mean_percentile"],
                s=18, alpha=alpha, color=color, marker=marker,
                label=f"{label}, {'correct' if correct else 'wrong'}",
            )
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Text mean catalog percentile")
    ax.set_ylabel("Image mean catalog percentile")
    ax.set_title("Mean-percentile decision geometry\n(Post-hoc exploratory)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    figure_files += save_figure(fig, "day18_mean_percentile_scatter")

    counts = [margin_bins[label]["samples"] for label in MARGIN_LABELS]
    accuracies = [margin_bins[label]["accuracy"] for label in MARGIN_LABELS]
    fig, ax1 = plt.subplots(figsize=(6.4, 3.8))
    positions = np.arange(len(MARGIN_LABELS))
    ax1.bar(positions, counts, color="#BBBBBB", label="Samples")
    ax1.set_ylabel("Samples")
    ax1.set_xticks(positions, MARGIN_LABELS)
    ax1.set_xlabel("Absolute image-text percentile margin")
    ax2 = ax1.twinx()
    ax2.plot(positions, accuracies, color="#AA3377", marker="o", linewidth=2, label="Accuracy")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Accuracy")
    ax1.set_title("Decision margin and correctness\n(Post-hoc exploratory)")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left")
    figure_files += save_figure(fig, "day18_margin_diagnostic")

    summary = {
        "status": "POSTHOC_EXPLORATORY_COMPLETE",
        "protocol_commit": PROTOCOL_COMMIT,
        "day17_metric_reproduction": observed,
        "margin_bins": margin_bins,
        "mean_max_agreement": {
            "counts": dict(Counter(frame["mean_max_group"])),
            "groups": mean_max,
        },
        "mean_vs_kimi": comparison,
        "deterministic_cases": cases,
        "figures": figure_files,
        "integrity": integrity,
        "inputs": {
            "day17_sha256": sha256(DAY17),
            "samples_sha256": sha256(SAMPLES),
            "blind_sha256": sha256(BLIND),
            "kimi_sha256": sha256(KIMI_CONF),
        },
    }
    serialized = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    OUTPUT.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
