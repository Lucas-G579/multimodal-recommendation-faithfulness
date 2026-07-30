"""Generate pair-level MGCN modality sensitivity records on the full test set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import verify_mgcn_checkpoint as runtime


PROJECT_ROOT = runtime.runtime.PROJECT_ROOT
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "audits" / "mgcn_behavior_audit.csv"
DEFAULT_SUMMARY = (
    PROJECT_ROOT / "outputs" / "audits" / "mgcn_behavior_audit_summary.json"
)
INTERACTIONS = PROJECT_ROOT / "external" / "MMRec" / "data" / "baby" / "baby.inter"


def zero_output(
    _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
) -> torch.Tensor:
    return torch.zeros_like(output)


@torch.no_grad()
def condition_embeddings(
    model: torch.nn.Module, image_zero: bool = False, text_zero: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    handles = []
    if image_zero:
        handles.append(model.gate_v.register_forward_hook(zero_output))
    if text_zero:
        handles.append(model.gate_t.register_forward_hook(zero_output))
    try:
        users, items = model.forward(model.norm_adj)
        return users.detach().cpu(), items.detach().cpu()
    finally:
        for handle in handles:
            handle.remove()


def move_inference_tensors(
    model: torch.nn.Module, device: torch.device
) -> torch.nn.Module:
    model = model.to(device)
    for attribute in ("norm_adj", "R", "image_original_adj", "text_original_adj"):
        tensor = getattr(model, attribute)
        setattr(model, attribute, tensor.to(device))
    return model


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def summarize_condition(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    score_delta = frame[f"{prefix}_score_drop"]
    rank_delta = frame[f"{prefix}_rank_change"]
    return {
        "score_drop": {
            "mean": float(score_delta.mean()),
            "median": float(score_delta.median()),
            "std": float(score_delta.std()),
            "q05": float(score_delta.quantile(0.05)),
            "q95": float(score_delta.quantile(0.95)),
            "positive_rate": float((score_delta > 0).mean()),
            "negative_rate": float((score_delta < 0).mean()),
        },
        "rank_change": {
            "mean": float(rank_delta.mean()),
            "median": float(rank_delta.median()),
            "q05": float(rank_delta.quantile(0.05)),
            "q95": float(rank_delta.quantile(0.95)),
            "worsened_rate": float((rank_delta > 0).mean()),
            "improved_rate": float((rank_delta < 0).mean()),
            "unchanged_rate": float((rank_delta == 0).mean()),
        },
        "top20": {
            "retained_rate_among_baseline_top20": float(
                frame.loc[frame["baseline_top20"], f"{prefix}_top20"].mean()
            ),
            "flip_out_count": int(
                (frame["baseline_top20"] & ~frame[f"{prefix}_top20"]).sum()
            ),
            "flip_in_count": int(
                (~frame["baseline_top20"] & frame[f"{prefix}_top20"]).sum()
            ),
        },
    }


def ranking_metrics_at_k(
    frame: pd.DataFrame, rank_column: str, k: int
) -> dict[str, float]:
    working = frame[["user_id", rank_column]].copy()
    working["hit"] = working[rank_column] <= k
    recall_by_user = working.groupby("user_id", sort=False)["hit"].mean()

    hit_ranks = working.loc[working["hit"], ["user_id", rank_column]].copy()
    hit_ranks["gain"] = 1.0 / np.log2(hit_ranks[rank_column] + 1.0)
    dcg = hit_ranks.groupby("user_id", sort=False)["gain"].sum()
    positive_counts = working.groupby("user_id", sort=False).size()
    ideal = positive_counts.map(
        lambda count: float(
            np.sum(1.0 / np.log2(np.arange(2, min(int(count), k) + 2)))
        )
    )
    ndcg_by_user = dcg.reindex(positive_counts.index, fill_value=0.0) / ideal
    return {
        f"recall@{k}": float(recall_by_user.mean()),
        f"ndcg@{k}": float(ndcg_by_user.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=runtime.DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--user-batch-size", type=int, default=256)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--numerical-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--inference-device", choices=["cpu", "cuda"], default="cpu"
    )
    args = parser.parse_args()

    model, config, _valid_data, _test_data = runtime.load_mgcn(
        args.checkpoint.resolve()
    )
    audit_device = torch.device(args.inference_device)
    model = move_inference_tensors(model, audit_device)
    model.eval()

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

    baseline_u, baseline_i = condition_embeddings(model)
    repeat_u, repeat_i = condition_embeddings(model)
    image_u, image_i = condition_embeddings(model, image_zero=True)
    text_u, text_i = condition_embeddings(model, text_zero=True)
    both_u, both_i = condition_embeddings(
        model, image_zero=True, text_zero=True
    )
    null_embedding_difference = max(
        float((baseline_u - repeat_u).abs().max()),
        float((baseline_i - repeat_i).abs().max()),
    )

    conditions = {
        "baseline": (baseline_u, baseline_i),
        "image_zero": (image_u, image_i),
        "text_zero": (text_u, text_i),
        "both_zero": (both_u, both_i),
    }
    device = audit_device
    conditions = {
        name: (user_embeddings.to(device), item_embeddings.to(device))
        for name, (user_embeddings, item_embeddings) in conditions.items()
    }
    users = test["userID"].drop_duplicates().to_numpy(dtype=np.int64)
    row_results: dict[str, dict[str, np.ndarray]] = {
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
        local_lookup = {int(user): idx for idx, user in enumerate(batch_users)}
        pair_local_users = np.fromiter(
            (local_lookup[int(user)] for user in pair_users),
            dtype=np.int64,
            count=len(pair_users),
        )

        for name, (user_embeddings, item_embeddings) in conditions.items():
            scores = torch.matmul(
                user_embeddings[batch_users],
                item_embeddings.T,
            )
            for local_user, user in enumerate(batch_users):
                seen = train_items.get(int(user))
                if seen is not None:
                    scores[local_user, torch.as_tensor(seen, device=device)] = -1e10

            pair_local_tensor = torch.as_tensor(pair_local_users, device=device)
            pair_item_tensor = torch.as_tensor(pair_items, device=device)
            target_scores = scores[pair_local_tensor, pair_item_tensor]
            target_ranks = 1 + (
                scores[pair_local_tensor] > target_scores.unsqueeze(1)
            ).sum(dim=1)
            row_results[name]["score"][row_indices] = (
                target_scores.detach().cpu().numpy()
            )
            row_results[name]["rank"][row_indices] = (
                target_ranks.detach().cpu().numpy()
            )

    audit = test.rename(columns={"userID": "user_id", "itemID": "item_id"})
    for name, values in row_results.items():
        audit[f"{name}_score"] = values["score"]
        audit[f"{name}_rank"] = values["rank"]
        audit[f"{name}_top20"] = values["rank"] <= args.topk

    for name in ("image_zero", "text_zero", "both_zero"):
        audit[f"{name}_score_drop"] = (
            audit["baseline_score"] - audit[f"{name}_score"]
        )
        audit[f"{name}_rank_change"] = (
            audit[f"{name}_rank"] - audit["baseline_rank"]
        )
    audit["score_drop_nonadditivity"] = (
        audit["both_zero_score_drop"]
        - audit["image_zero_score_drop"]
        - audit["text_zero_score_drop"]
    )

    image_magnitude = audit["image_zero_score_drop"].abs()
    text_magnitude = audit["text_zero_score_drop"].abs()
    magnitude_sum = image_magnitude + text_magnitude
    audit["image_sensitivity_share"] = np.where(
        magnitude_sum > args.numerical_tolerance,
        image_magnitude / magnitude_sum,
        0.5,
    )
    audit["text_sensitivity_share"] = 1.0 - audit["image_sensitivity_share"]
    audit["larger_abs_score_change_modality"] = np.select(
        [
            image_magnitude - text_magnitude > args.numerical_tolerance,
            text_magnitude - image_magnitude > args.numerical_tolerance,
        ],
        ["image", "text"],
        default="tie",
    )

    required_numeric = audit.select_dtypes(include=[np.number])
    if not np.isfinite(required_numeric.to_numpy()).all():
        raise ValueError("Audit contains NaN or infinite numeric values")
    if len(audit) != len(test) or audit.duplicated(["user_id", "item_id"]).any():
        raise ValueError("Audit row identity check failed")
    rank_columns = [column for column in audit if column.endswith("_rank")]
    if audit[rank_columns].min().min() < 1:
        raise ValueError("Ranks must be one-based positive integers")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output, index=False, float_format="%.9g")
    output_hash = sha256_file(args.output)

    summary = {
        "status": "PASSED",
        "checkpoint_sha256": runtime.runtime.sha256_file(
            args.checkpoint.resolve()
        ),
        "audit_sha256": output_hash,
        "scope": {
            "test_pairs": int(len(audit)),
            "test_users": int(audit["user_id"].nunique()),
            "items": int(model.n_items),
            "user_batch_size": args.user_batch_size,
            "inference_device": str(audit_device),
        },
        "protocol": {
            "candidate_mask": "training interactions only",
            "rank_tie_rule": "optimistic: 1 + count(scores > target_score)",
            "intervention": (
                "global zero ablation of the selected MGCN behavior-guided "
                "modality gate output"
            ),
            "numerical_tolerance": args.numerical_tolerance,
            "null_max_embedding_difference": null_embedding_difference,
        },
        "baseline": {
            "top20_pair_rate": float(audit["baseline_top20"].mean()),
            "median_rank": float(audit["baseline_rank"].median()),
            "ranking_metrics": ranking_metrics_at_k(
                audit, "baseline_rank", args.topk
            ),
        },
        "image_zero": summarize_condition(audit, "image_zero")
        | {
            "ranking_metrics": ranking_metrics_at_k(
                audit, "image_zero_rank", args.topk
            )
        },
        "text_zero": summarize_condition(audit, "text_zero")
        | {
            "ranking_metrics": ranking_metrics_at_k(
                audit, "text_zero_rank", args.topk
            )
        },
        "both_zero": summarize_condition(audit, "both_zero")
        | {
            "ranking_metrics": ranking_metrics_at_k(
                audit, "both_zero_rank", args.topk
            )
        },
        "larger_abs_score_change_counts": {
            str(key): int(value)
            for key, value in audit[
                "larger_abs_score_change_modality"
            ].value_counts().items()
        },
        "interaction_check": {
            "score_drop_nonadditivity_mean": float(
                audit["score_drop_nonadditivity"].mean()
            ),
            "score_drop_nonadditivity_mean_absolute": float(
                audit["score_drop_nonadditivity"].abs().mean()
            ),
        },
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
