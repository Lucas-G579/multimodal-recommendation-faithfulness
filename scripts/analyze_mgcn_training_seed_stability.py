"""Intersect strict MGCN modality labels across training seeds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = PROJECT_ROOT / "outputs" / "audits"
SEEDS = (999, 2026, 3407)
LABEL_PATHS = {
    999: AUDIT_ROOT / "mgcn_high_confidence_labels.csv",
    2026: AUDIT_ROOT / "mgcn_seed_2026_high_confidence_labels.csv",
    3407: AUDIT_ROOT / "mgcn_seed_3407_high_confidence_labels.csv",
}
SUMMARY_PATHS = {
    999: AUDIT_ROOT / "mgcn_permutation_seed_stability_summary.json",
    2026: AUDIT_ROOT / "mgcn_seed_2026_permutation_seed_stability_summary.json",
    3407: AUDIT_ROOT / "mgcn_seed_3407_permutation_seed_stability_summary.json",
}
OUTPUT = AUDIT_ROOT / "mgcn_cross_training_seed_labels.csv"
SUMMARY_OUTPUT = AUDIT_ROOT / "mgcn_cross_training_seed_stability_summary.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def comparable_agreement(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    comparable = (first != "unstable") & (second != "unstable")
    return {
        "comparable_A_pairs": int(comparable.sum()),
        "agreement": (
            float((first[comparable] == second[comparable]).mean())
            if comparable.any()
            else None
        ),
    }


def counts(values: np.ndarray) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in pd.Series(values).value_counts().items()
    }


def main() -> None:
    frames: dict[int, pd.DataFrame] = {}
    input_metadata: dict[str, Any] = {}
    required = {
        "user_id",
        "item_id",
        "strict_high_confidence_label",
        "confidence_tier",
        "tiered_modality_label",
    }
    for seed in SEEDS:
        label_path = LABEL_PATHS[seed]
        summary_path = SUMMARY_PATHS[seed]
        frame = pd.read_csv(label_path)
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"seed={seed} labels missing columns: {sorted(missing)}")
        if frame.duplicated(["user_id", "item_id"]).any():
            raise ValueError(f"seed={seed} contains duplicate pairs")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        label_hash = sha256_file(label_path)
        if label_hash != summary["labels_sha256"]:
            raise ValueError(f"seed={seed} label hash does not match summary")
        frames[seed] = frame
        input_metadata[str(seed)] = {
            "labels_path": str(label_path),
            "labels_sha256": label_hash,
            "summary_path": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "checkpoint_sha256": summary["checkpoint_sha256"],
        }

    reference_pairs = frames[SEEDS[0]][["user_id", "item_id"]].to_numpy()
    for seed in SEEDS[1:]:
        if not np.array_equal(
            reference_pairs,
            frames[seed][["user_id", "item_id"]].to_numpy(),
        ):
            raise ValueError(f"Pair identity/order differs for seed={seed}")

    output = frames[SEEDS[0]][["user_id", "item_id"]].copy()
    strict_matrix = []
    tiered_matrix = []
    for seed in SEEDS:
        strict = frames[seed]["strict_high_confidence_label"].to_numpy()
        tiered = frames[seed]["tiered_modality_label"].to_numpy()
        output[f"seed_{seed}_strict_label"] = strict
        output[f"seed_{seed}_confidence_tier"] = frames[seed][
            "confidence_tier"
        ].to_numpy()
        output[f"seed_{seed}_tiered_label"] = tiered
        strict_matrix.append(strict)
        tiered_matrix.append(tiered)

    strict_matrix_array = np.column_stack(strict_matrix)
    tiered_matrix_array = np.column_stack(tiered_matrix)
    strict_first = strict_matrix_array[:, 0]
    tiered_first = tiered_matrix_array[:, 0]
    cross_seed_A = (
        np.all(strict_matrix_array == strict_first[:, None], axis=1)
        & (strict_first != "unstable")
    )
    cross_seed_A_or_B = (
        np.all(tiered_matrix_array == tiered_first[:, None], axis=1)
        & (tiered_first != "unstable")
    )
    output["cross_seed_A_label"] = np.where(
        cross_seed_A, strict_first, "unstable"
    )
    output["cross_seed_A_or_B_label"] = np.where(
        cross_seed_A_or_B, tiered_first, "unstable"
    )

    pairwise: dict[str, Any] = {}
    for first_index, first_seed in enumerate(SEEDS):
        for second_seed in SEEDS[first_index + 1 :]:
            pairwise[f"{first_seed}_vs_{second_seed}"] = comparable_agreement(
                frames[first_seed]["strict_high_confidence_label"].to_numpy(),
                frames[second_seed]["strict_high_confidence_label"].to_numpy(),
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False)
    cross_A_labels = output["cross_seed_A_label"].to_numpy()
    cross_AB_labels = output["cross_seed_A_or_B_label"].to_numpy()
    summary = {
        "status": "PASSED",
        "scope": {
            "training_seeds": list(SEEDS),
            "pairs": int(len(output)),
        },
        "protocol": {
            "primary_rule": (
                "A in every training seed and identical non-unstable modality"
            ),
            "sensitivity_rule": (
                "A or B in every training seed and identical non-unstable modality"
            ),
        },
        "inputs": input_metadata,
        "per_training_seed_A_counts": {
            str(seed): counts(
                frames[seed]["strict_high_confidence_label"].to_numpy()
            )
            for seed in SEEDS
        },
        "pairwise_A_agreement": pairwise,
        "cross_seed_A": {
            "coverage": float(cross_seed_A.mean()),
            "counts": counts(cross_A_labels),
            "retention_from_seed_999_A": float(
                cross_seed_A.sum()
                / (
                    frames[999]["strict_high_confidence_label"].to_numpy()
                    != "unstable"
                ).sum()
            ),
        },
        "cross_seed_A_or_B": {
            "coverage": float(cross_seed_A_or_B.mean()),
            "counts": counts(cross_AB_labels),
        },
    }
    summary["output_sha256"] = sha256_file(OUTPUT)
    SUMMARY_OUTPUT.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
