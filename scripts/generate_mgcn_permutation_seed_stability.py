"""Evaluate MGCN modality labels across multiple fixed permutation seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import mgcn_intervention_runtime as audit_forward
import verify_mgcn_checkpoint as runtime
from generate_mgcn_behavior_audit import move_inference_tensors
from generate_mgcn_intervention_robustness import (
    label_from_rank_changes,
    pairwise_agreement,
    ranking_metrics,
)


PROJECT_ROOT = runtime.runtime.PROJECT_ROOT
INTERACTIONS = PROJECT_ROOT / "external" / "MMRec" / "data" / "baby" / "baby.inter"
ROBUSTNESS_AUDIT = (
    PROJECT_ROOT / "outputs" / "audits" / "mgcn_intervention_robustness.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "audits"
    / "mgcn_permutation_seed_stability.csv"
)
DEFAULT_LABELS = (
    PROJECT_ROOT / "outputs" / "audits" / "mgcn_high_confidence_labels.csv"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "audits"
    / "mgcn_permutation_seed_stability_summary.json"
)
DEFAULT_SEEDS = (20260730, 20260731, 20260732, 20260733, 20260734)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if len(seeds) < 2:
        raise ValueError("At least two permutation seeds are required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Permutation seeds must be unique")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=runtime.DEFAULT_CHECKPOINT)
    parser.add_argument("--robustness-audit", type=Path, default=ROBUSTNESS_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--labels-output", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--user-batch-size", type=int, default=256)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated, pre-specified permutation seeds",
    )
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)

    reference = pd.read_csv(args.robustness_audit)
    required = {
        "user_id",
        "item_id",
        "baseline_rank",
        "zero_larger_abs_rank_change",
        "mean_larger_abs_rank_change",
    }
    missing = required.difference(reference.columns)
    if missing:
        raise KeyError(f"Robustness audit missing columns: {sorted(missing)}")

    model, _config, _valid_data, _test_data = runtime.load_mgcn(
        args.checkpoint.resolve()
    )
    model = move_inference_tensors(model, torch.device("cpu"))
    model.eval()
    components = audit_forward.decompose(model)

    conditions: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for seed in seeds:
        conditions[f"image_{seed}"] = audit_forward.fuse(
            model,
            components,
            image_mode="permutation",
            permutation_seed=seed,
        )
        conditions[f"text_{seed}"] = audit_forward.fuse(
            model,
            components,
            text_mode="permutation",
            permutation_seed=seed,
        )

    interactions = pd.read_csv(INTERACTIONS, sep="\t")
    train = interactions.loc[interactions["x_label"] == 0, ["userID", "itemID"]]
    test = (
        interactions.loc[interactions["x_label"] == 2, ["userID", "itemID"]]
        .sort_values(["userID", "itemID"], kind="stable")
        .reset_index(drop=True)
    )
    if not np.array_equal(
        reference[["user_id", "item_id"]].to_numpy(),
        test[["userID", "itemID"]].to_numpy(),
    ):
        raise ValueError("Reference and test pair order differ")

    train_items = {
        int(user): group["itemID"].to_numpy(dtype=np.int64)
        for user, group in train.groupby("userID", sort=False)
    }
    users = test["userID"].drop_duplicates().to_numpy(dtype=np.int64)
    ranks = {
        name: np.empty(len(test), dtype=np.int32) for name in conditions
    }

    for start in range(0, len(users), args.user_batch_size):
        batch_users = users[start : start + args.user_batch_size]
        row_mask = test["userID"].isin(batch_users).to_numpy()
        row_indices = np.flatnonzero(row_mask)
        pair_users = test.loc[row_mask, "userID"].to_numpy(dtype=np.int64)
        pair_items = test.loc[row_mask, "itemID"].to_numpy(dtype=np.int64)
        local = {int(user): index for index, user in enumerate(batch_users)}
        pair_local = np.fromiter(
            (local[int(user)] for user in pair_users),
            dtype=np.int64,
            count=len(pair_users),
        )
        pair_local_tensor = torch.from_numpy(pair_local)
        pair_item_tensor = torch.from_numpy(pair_items)

        for name, (user_embeddings, item_embeddings) in conditions.items():
            scores = torch.matmul(
                user_embeddings[batch_users], item_embeddings.T
            )
            for local_user, user in enumerate(batch_users):
                seen = train_items.get(int(user))
                if seen is not None:
                    scores[local_user, torch.from_numpy(seen)] = -1e10
            targets = scores[pair_local_tensor, pair_item_tensor]
            target_ranks = 1 + (
                scores[pair_local_tensor] > targets.unsqueeze(1)
            ).sum(dim=1)
            ranks[name][row_indices] = target_ranks.numpy()

    long_frames = []
    seed_labels: dict[int, np.ndarray] = {}
    metrics_by_seed: dict[str, dict[str, dict[str, float]]] = {}
    for seed in seeds:
        seed_frame = reference[
            ["user_id", "item_id", "baseline_rank"]
        ].copy()
        seed_frame.insert(2, "permutation_seed", seed)
        seed_frame["image_permutation_rank"] = ranks[f"image_{seed}"]
        seed_frame["text_permutation_rank"] = ranks[f"text_{seed}"]
        seed_frame["image_rank_change"] = (
            seed_frame["image_permutation_rank"] - seed_frame["baseline_rank"]
        )
        seed_frame["text_rank_change"] = (
            seed_frame["text_permutation_rank"] - seed_frame["baseline_rank"]
        )
        labels = label_from_rank_changes(
            seed_frame["image_rank_change"], seed_frame["text_rank_change"]
        )
        seed_frame["larger_abs_rank_change_modality"] = labels
        seed_labels[seed] = labels
        metrics_by_seed[str(seed)] = {
            "image": ranking_metrics(
                seed_frame, "image_permutation_rank", args.topk
            ),
            "text": ranking_metrics(
                seed_frame, "text_permutation_rank", args.topk
            ),
        }
        long_frames.append(seed_frame)

    long_audit = pd.concat(long_frames, ignore_index=True)
    label_matrix = np.column_stack([seed_labels[seed] for seed in seeds])
    first = label_matrix[:, 0]
    permutation_unanimous = (
        np.all(label_matrix == first[:, None], axis=1) & (first != "tie")
    )
    image_votes = (label_matrix == "image").sum(axis=1)
    text_votes = (label_matrix == "text").sum(axis=1)
    tie_votes = (label_matrix == "tie").sum(axis=1)
    majority_label = np.select(
        [image_votes > text_votes, text_votes > image_votes],
        ["image", "text"],
        default="tie",
    )
    majority_fraction = np.maximum(image_votes, text_votes) / len(seeds)

    zero_labels = reference["zero_larger_abs_rank_change"].to_numpy()
    mean_labels = reference["mean_larger_abs_rank_change"].to_numpy()
    strict = (
        permutation_unanimous
        & (zero_labels == mean_labels)
        & (zero_labels == first)
        & (zero_labels != "tie")
    )
    final_labels = np.where(strict, zero_labels, "unstable")
    tier_b_or_better = (
        (zero_labels == mean_labels)
        & (zero_labels == majority_label)
        & (zero_labels != "tie")
        & (majority_fraction >= 0.8)
    )
    confidence_tier = np.select(
        [strict, tier_b_or_better],
        ["A", "B"],
        default="unstable",
    )
    tiered_label = np.where(tier_b_or_better, zero_labels, "unstable")
    labels_output = reference[["user_id", "item_id"]].copy()
    labels_output["zero_label"] = zero_labels
    labels_output["mean_label"] = mean_labels
    labels_output["permutation_unanimous_label"] = np.where(
        permutation_unanimous, first, "unstable"
    )
    labels_output["permutation_majority_label"] = majority_label
    labels_output["permutation_majority_fraction"] = majority_fraction
    labels_output["permutation_tie_votes"] = tie_votes
    labels_output["strict_high_confidence_label"] = final_labels
    labels_output["confidence_tier"] = confidence_tier
    labels_output["tiered_modality_label"] = tiered_label

    pairwise: dict[str, Any] = {}
    for first_index, first_seed in enumerate(seeds):
        for second_seed in seeds[first_index + 1 :]:
            pairwise[f"{first_seed}_vs_{second_seed}"] = pairwise_agreement(
                seed_labels[first_seed], seed_labels[second_seed]
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    long_audit.to_csv(args.output, index=False)
    labels_output.to_csv(args.labels_output, index=False)
    summary = {
        "status": "PASSED",
        "checkpoint_sha256": runtime.runtime.sha256_file(
            args.checkpoint.resolve()
        ),
        "source_audit_sha256": sha256_file(args.robustness_audit),
        "long_audit_sha256": sha256_file(args.output),
        "labels_sha256": sha256_file(args.labels_output),
        "scope": {
            "seeds": list(seeds),
            "seed_count": len(seeds),
            "pairs": int(len(reference)),
            "long_rows": int(len(long_audit)),
        },
        "protocol": {
            "seed_selection": "pre-specified consecutive seeds; no result-based selection",
            "strict_label_rule": (
                "zero, mean, and every permutation seed must agree on the "
                "same non-tie modality"
            ),
        },
        "ranking_metrics_by_seed": metrics_by_seed,
        "label_counts_by_seed": {
            str(seed): {
                str(key): int(value)
                for key, value in pd.Series(seed_labels[seed]).value_counts().items()
            }
            for seed in seeds
        },
        "permutation_unanimous": {
            "coverage": float(permutation_unanimous.mean()),
            "counts": {
                str(key): int(value)
                for key, value in pd.Series(
                    np.where(permutation_unanimous, first, "unstable")
                ).value_counts().items()
            },
        },
        "pairwise_seed_agreement": pairwise,
        "strict_high_confidence": {
            "coverage": float(strict.mean()),
            "counts": {
                str(key): int(value)
                for key, value in pd.Series(final_labels).value_counts().items()
            },
        },
        "tiered_labels": {
            "definition": {
                "A": "zero, mean, and all 5 permutation seeds agree",
                "B": (
                    "zero and mean agree, and at least 4 of 5 permutation "
                    "seeds agree with them"
                ),
                "unstable": "all remaining pairs",
            },
            "tier_counts": {
                str(key): int(value)
                for key, value in pd.Series(confidence_tier).value_counts().items()
            },
            "modality_counts_for_A_or_B": {
                str(key): int(value)
                for key, value in pd.Series(tiered_label).value_counts().items()
            },
        },
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
