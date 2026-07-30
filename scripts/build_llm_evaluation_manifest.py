"""Freeze user-disjoint LLM evaluation cohorts before inspecting metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = PROJECT_ROOT / "outputs" / "audits"
LABELS = AUDIT_ROOT / "mgcn_cross_training_seed_labels.csv"
CONTINUOUS = AUDIT_ROOT / "mgcn_cross_training_seed_continuous_sensitivity.csv"
OUTPUT = PROJECT_ROOT / "data" / "manifests" / "llm_evaluation_samples.csv"
SUMMARY = PROJECT_ROOT / "data" / "manifests" / "llm_evaluation_samples.json"
HASH_SALT = "faithrec-mm-evaluation-v1"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def hash_value(user_id: int, item_id: int, purpose: str) -> str:
    value = f"{HASH_SALT}:{purpose}:{user_id}:{item_id}".encode()
    return hashlib.sha256(value).hexdigest()


def select_user_unique(
    frame: pd.DataFrame, count: int, purpose: str
) -> pd.DataFrame:
    working = frame.copy()
    working["_selection_hash"] = [
        hash_value(user, item, purpose)
        for user, item in zip(working["user_id"], working["item_id"])
    ]
    working.sort_values("_selection_hash", inplace=True)
    working.drop_duplicates("user_id", keep="first", inplace=True)
    if len(working) < count:
        raise ValueError(
            f"Only {len(working)} user-unique candidates for {purpose}, need {count}"
        )
    return working.head(count).drop(columns="_selection_hash")


def main() -> None:
    labels = pd.read_csv(LABELS)
    continuous = pd.read_csv(CONTINUOUS)
    if not labels[["user_id", "item_id"]].equals(
        continuous[["user_id", "item_id"]]
    ):
        raise ValueError("Label and continuous pair order differ")
    data = labels.merge(
        continuous,
        on=["user_id", "item_id"],
        how="inner",
        validate="one_to_one",
    )

    strict = data.loc[data["cross_seed_A_label"] != "unstable"].copy()
    strict["user_bucket"] = strict["user_id"].map(
        lambda user: int(
            hashlib.sha256(f"{HASH_SALT}:dev-user:{user}".encode()).hexdigest()[:8],
            16,
        )
        % 10
    )
    strict["cohort"] = "primary_confirmatory"
    strict.loc[strict["user_bucket"] == 0, "cohort"] = "prompt_development"

    strict_users = set(strict["user_id"])
    sensitivity_candidates = data.loc[
        (data["cross_seed_A_label"] == "unstable")
        & (data["cross_seed_A_or_B_label"] != "unstable")
        & (~data["user_id"].isin(strict_users))
    ]
    sensitivity_parts = []
    for modality in ("image", "text"):
        subset = sensitivity_candidates.loc[
            sensitivity_candidates["cross_seed_A_or_B_label"] == modality
        ]
        selected = select_user_unique(
            subset, 200, f"sensitivity-{modality}"
        )
        selected["cohort"] = "A_or_B_sensitivity"
        sensitivity_parts.append(selected)
    sensitivity = pd.concat(sensitivity_parts, ignore_index=True)

    excluded_users = strict_users | set(sensitivity["user_id"])
    unstable_candidates = data.loc[
        (data["cross_seed_A_or_B_label"] == "unstable")
        & (~data["user_id"].isin(excluded_users))
    ]
    unstable = select_user_unique(
        unstable_candidates, 400, "unstable-overconfidence"
    )
    unstable["cohort"] = "unstable_overconfidence"

    manifest = pd.concat([strict, sensitivity, unstable], ignore_index=True)
    role_order = {
        "prompt_development": 0,
        "primary_confirmatory": 1,
        "A_or_B_sensitivity": 2,
        "unstable_overconfidence": 3,
    }
    manifest["_cohort_order"] = manifest["cohort"].map(role_order)
    manifest.sort_values(
        ["_cohort_order", "user_id", "item_id"], inplace=True
    )
    manifest.drop(columns="_cohort_order", inplace=True)
    manifest.insert(
        0,
        "sample_id",
        [
            f"baby-u{user}-i{item}"
            for user, item in zip(manifest["user_id"], manifest["item_id"])
        ],
    )

    user_cohort_counts = manifest.groupby("user_id")["cohort"].nunique()
    if (user_cohort_counts > 1).any():
        raise ValueError("A user appears in multiple evaluation cohorts")
    if manifest.duplicated(["user_id", "item_id"]).any():
        raise ValueError("Duplicate evaluation pairs")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT, index=False, float_format="%.9g")
    cohort_counts = {
        str(cohort): int(value)
        for cohort, value in manifest["cohort"].value_counts().items()
    }
    label_counts = {
        cohort: {
            "strict_A": {
                str(key): int(value)
                for key, value in group["cross_seed_A_label"].value_counts().items()
            },
            "A_or_B": {
                str(key): int(value)
                for key, value in group[
                    "cross_seed_A_or_B_label"
                ].value_counts().items()
            },
        }
        for cohort, group in manifest.groupby("cohort", sort=False)
    }
    summary = {
        "status": "FROZEN",
        "hash_salt": HASH_SALT,
        "selection": {
            "prompt_development": (
                "all strict-A pairs whose SHA256 user bucket modulo 10 equals 0"
            ),
            "primary_confirmatory": (
                "all remaining strict-A pairs; no user overlaps development"
            ),
            "A_or_B_sensitivity": (
                "A-or-B but not A, excluding strict-A users; 200 user-unique "
                "pairs per modality by SHA256 order"
            ),
            "unstable_overconfidence": (
                "A-or-B unstable, excluding all prior users; 400 user-unique "
                "pairs by SHA256 order"
            ),
        },
        "rows": int(len(manifest)),
        "unique_users": int(manifest["user_id"].nunique()),
        "cohort_counts": cohort_counts,
        "label_counts": label_counts,
        "inputs": {
            "labels_sha256": sha256_file(LABELS),
            "continuous_sha256": sha256_file(CONTINUOUS),
        },
        "output_sha256": sha256_file(OUTPUT),
    }
    SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
