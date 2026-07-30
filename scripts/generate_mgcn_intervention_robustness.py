"""Compare zero, mean, and permutation interventions on MGCN modality branches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import mgcn_intervention_runtime as audit_forward
import verify_mgcn_checkpoint as runtime
from generate_mgcn_behavior_audit import move_inference_tensors


PROJECT_ROOT = runtime.runtime.PROJECT_ROOT
INTERACTIONS = PROJECT_ROOT / "external" / "MMRec" / "data" / "baby" / "baby.inter"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "audits" / "mgcn_intervention_robustness.csv"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "audits"
    / "mgcn_intervention_robustness_summary.json"
)
METHODS = ("zero", "mean", "permutation")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def ranking_metrics(frame: pd.DataFrame, rank_column: str, topk: int) -> dict[str, float]:
    working = frame[["user_id", rank_column]].copy()
    working["hit"] = working[rank_column] <= topk
    working["gain"] = np.where(
        working["hit"], 1.0 / np.log2(working[rank_column] + 1.0), 0.0
    )
    grouped = working.groupby("user_id", sort=False)
    recall = grouped["hit"].mean().mean()
    dcg = grouped["gain"].sum()
    positives = grouped.size()
    idcg = positives.map(
        lambda count: float(
            np.sum(1.0 / np.log2(np.arange(2, min(int(count), topk) + 2)))
        )
    )
    return {
        f"recall@{topk}": float(recall),
        f"ndcg@{topk}": float((dcg / idcg).mean()),
    }


def label_from_rank_changes(
    image: pd.Series, text: pd.Series
) -> np.ndarray:
    image_abs = image.abs().to_numpy()
    text_abs = text.abs().to_numpy()
    return np.select(
        [image_abs > text_abs, text_abs > image_abs],
        ["image", "text"],
        default="tie",
    )


def pairwise_agreement(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    comparable = (first != "tie") & (second != "tie")
    return {
        "comparable_pairs": int(comparable.sum()),
        "agreement": (
            float((first[comparable] == second[comparable]).mean())
            if comparable.any()
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=runtime.DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--user-batch-size", type=int, default=256)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--permutation-seed", type=int, default=20260730)
    args = parser.parse_args()

    model, _config, _valid_data, _test_data = runtime.load_mgcn(
        args.checkpoint.resolve()
    )
    model = move_inference_tensors(model, torch.device("cpu"))
    model.eval()
    components = audit_forward.decompose(model)

    with torch.no_grad():
        original_users, original_items = model.forward(model.norm_adj)
    rebuilt_users, rebuilt_items = audit_forward.fuse(model, components)
    forward_equivalence = max(
        float((original_users - rebuilt_users).abs().max()),
        float((original_items - rebuilt_items).abs().max()),
    )
    if forward_equivalence != 0.0:
        raise ValueError(
            f"Audit forward does not exactly reproduce MGCN: {forward_equivalence}"
        )

    conditions: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        "baseline": (rebuilt_users, rebuilt_items)
    }
    for method in METHODS:
        conditions[f"image_{method}"] = audit_forward.fuse(
            model,
            components,
            image_mode=method,
            permutation_seed=args.permutation_seed,
        )
        conditions[f"text_{method}"] = audit_forward.fuse(
            model,
            components,
            text_mode=method,
            permutation_seed=args.permutation_seed,
        )

    interactions = pd.read_csv(INTERACTIONS, sep="\t")
    train = interactions.loc[interactions["x_label"] == 0, ["userID", "itemID"]]
    test = (
        interactions.loc[interactions["x_label"] == 2, ["userID", "itemID"]]
        .sort_values(["userID", "itemID"], kind="stable")
        .reset_index(drop=True)
    )
    train_items = {
        int(user): group["itemID"].to_numpy(dtype=np.int64)
        for user, group in train.groupby("userID", sort=False)
    }
    users = test["userID"].drop_duplicates().to_numpy(dtype=np.int64)
    results = {
        name: {
            "score": np.empty(len(test), dtype=np.float32),
            "rank": np.empty(len(test), dtype=np.int32),
        }
        for name in conditions
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
            ranks = 1 + (
                scores[pair_local_tensor] > targets.unsqueeze(1)
            ).sum(dim=1)
            results[name]["score"][row_indices] = targets.numpy()
            results[name]["rank"][row_indices] = ranks.numpy()

    audit = test.rename(columns={"userID": "user_id", "itemID": "item_id"})
    for name, values in results.items():
        audit[f"{name}_score"] = values["score"]
        audit[f"{name}_rank"] = values["rank"]
    for method in METHODS:
        for modality in ("image", "text"):
            prefix = f"{modality}_{method}"
            audit[f"{prefix}_score_drop"] = (
                audit["baseline_score"] - audit[f"{prefix}_score"]
            )
            audit[f"{prefix}_rank_change"] = (
                audit[f"{prefix}_rank"] - audit["baseline_rank"]
            )
        audit[f"{method}_larger_abs_rank_change"] = label_from_rank_changes(
            audit[f"image_{method}_rank_change"],
            audit[f"text_{method}_rank_change"],
        )

    labels = {
        method: audit[f"{method}_larger_abs_rank_change"].to_numpy()
        for method in METHODS
    }
    stable = (
        (labels["zero"] == labels["mean"])
        & (labels["zero"] == labels["permutation"])
        & (labels["zero"] != "tie")
    )
    audit["stable_modality_label"] = np.where(stable, labels["zero"], "unstable")

    if audit.duplicated(["user_id", "item_id"]).any():
        raise ValueError("Duplicate audit rows")
    if not np.isfinite(audit.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError("Audit contains NaN or Inf")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output, index=False, float_format="%.9g")
    summary: dict[str, Any] = {
        "status": "PASSED",
        "checkpoint_sha256": runtime.runtime.sha256_file(
            args.checkpoint.resolve()
        ),
        "audit_sha256": sha256_file(args.output),
        "scope": {
            "pairs": int(len(audit)),
            "users": int(audit["user_id"].nunique()),
            "intervention_methods": list(METHODS),
        },
        "protocol": {
            "inference_device": "cpu",
            "forward_equivalence_max_difference": forward_equivalence,
            "permutation_seed": args.permutation_seed,
            "intervention_locus": (
                "enriched modality-specific item representation after "
                "item-item graph propagation; user modality representation "
                "is recomputed from intervened items"
            ),
        },
        "ranking_metrics": {
            name: ranking_metrics(audit, f"{name}_rank", args.topk)
            for name in conditions
        },
        "label_counts": {
            method: {
                str(key): int(value)
                for key, value in audit[
                    f"{method}_larger_abs_rank_change"
                ].value_counts().items()
            }
            for method in METHODS
        },
        "pairwise_label_agreement": {
            "zero_vs_mean": pairwise_agreement(labels["zero"], labels["mean"]),
            "zero_vs_permutation": pairwise_agreement(
                labels["zero"], labels["permutation"]
            ),
            "mean_vs_permutation": pairwise_agreement(
                labels["mean"], labels["permutation"]
            ),
        },
        "stable_labels": {
            "coverage": float(stable.mean()),
            "counts": {
                str(key): int(value)
                for key, value in audit["stable_modality_label"]
                .value_counts()
                .items()
            },
        },
        "rank_sensitivity_spearman": {},
    }
    for modality in ("image", "text"):
        for first, second in (
            ("zero", "mean"),
            ("zero", "permutation"),
            ("mean", "permutation"),
        ):
            correlation = spearmanr(
                audit[f"{modality}_{first}_rank_change"].abs(),
                audit[f"{modality}_{second}_rank_change"].abs(),
            ).statistic
            summary["rank_sensitivity_spearman"][
                f"{modality}.{first}_vs_{second}"
            ] = float(correlation)

    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
