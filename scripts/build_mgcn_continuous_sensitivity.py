"""Build cross-training-seed continuous modality sensitivity measurements."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = PROJECT_ROOT / "outputs" / "audits"
TRAINING_SEEDS = (999, 2026, 3407)
PERMUTATION_SEEDS = (20260730, 20260731, 20260732, 20260733, 20260734)
OUTPUT = AUDIT_ROOT / "mgcn_cross_training_seed_continuous_sensitivity.csv"
SUMMARY_OUTPUT = (
    AUDIT_ROOT / "mgcn_cross_training_seed_continuous_sensitivity_summary.json"
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def paths_for(training_seed: int) -> tuple[Path, Path]:
    if training_seed == 999:
        return (
            AUDIT_ROOT / "mgcn_intervention_robustness.csv",
            AUDIT_ROOT / "mgcn_permutation_seed_stability.csv",
        )
    prefix = f"mgcn_seed_{training_seed}"
    return (
        AUDIT_ROOT / f"{prefix}_intervention_robustness.csv",
        AUDIT_ROOT / f"{prefix}_permutation_seed_stability.csv",
    )


def normalized_contrast(image_change: np.ndarray, text_change: np.ndarray) -> np.ndarray:
    image_abs = np.abs(image_change.astype(np.float64))
    text_abs = np.abs(text_change.astype(np.float64))
    denominator = image_abs + text_abs
    return np.divide(
        text_abs - image_abs,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )


def main() -> None:
    keys: pd.DataFrame | None = None
    measurements: dict[str, np.ndarray] = {}
    input_hashes: dict[str, str] = {}

    for training_seed in TRAINING_SEEDS:
        robustness_path, permutation_path = paths_for(training_seed)
        robustness = pd.read_csv(robustness_path)
        permutation = pd.read_csv(permutation_path)
        current_keys = robustness[["user_id", "item_id"]]
        if keys is None:
            keys = current_keys.copy()
        elif not np.array_equal(keys.to_numpy(), current_keys.to_numpy()):
            raise ValueError(f"Robustness pair order differs for seed={training_seed}")
        if robustness.duplicated(["user_id", "item_id"]).any():
            raise ValueError(f"Duplicate robustness pairs for seed={training_seed}")
        if len(permutation) != len(robustness) * len(PERMUTATION_SEEDS):
            raise ValueError(f"Unexpected permutation row count for seed={training_seed}")

        input_hashes[str(robustness_path)] = sha256_file(robustness_path)
        input_hashes[str(permutation_path)] = sha256_file(permutation_path)
        for method in ("zero", "mean"):
            name = f"train_{training_seed}_{method}"
            measurements[name] = normalized_contrast(
                robustness[f"image_{method}_rank_change"].to_numpy(),
                robustness[f"text_{method}_rank_change"].to_numpy(),
            )
        for permutation_seed in PERMUTATION_SEEDS:
            subset = permutation.loc[
                permutation["permutation_seed"] == permutation_seed
            ]
            if not np.array_equal(
                keys.to_numpy(),
                subset[["user_id", "item_id"]].to_numpy(),
            ):
                raise ValueError(
                    "Permutation pair order differs for "
                    f"training_seed={training_seed}, permutation_seed={permutation_seed}"
                )
            name = f"train_{training_seed}_perm_{permutation_seed}"
            measurements[name] = normalized_contrast(
                subset["image_rank_change"].to_numpy(),
                subset["text_rank_change"].to_numpy(),
            )

    assert keys is not None
    matrix = np.column_stack(list(measurements.values()))
    output = keys.copy()
    output["contrast_median"] = np.median(matrix, axis=1)
    output["contrast_mean"] = matrix.mean(axis=1)
    output["contrast_mad"] = np.median(
        np.abs(matrix - output["contrast_median"].to_numpy()[:, None]),
        axis=1,
    )
    positive = (matrix > 0).sum(axis=1)
    negative = (matrix < 0).sum(axis=1)
    zero = (matrix == 0).sum(axis=1)
    output["text_direction_votes"] = positive
    output["image_direction_votes"] = negative
    output["tie_votes"] = zero
    output["majority_direction_fraction"] = np.maximum(positive, negative) / matrix.shape[1]
    output["continuous_direction"] = np.select(
        [output["contrast_median"] > 0, output["contrast_median"] < 0],
        ["text", "image"],
        default="tie",
    )
    output["measurements"] = matrix.shape[1]

    if not np.isfinite(output.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError("Continuous sensitivity output contains NaN or Inf")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False, float_format="%.9g")
    summary = {
        "status": "PASSED",
        "scope": {
            "pairs": int(len(output)),
            "training_seeds": list(TRAINING_SEEDS),
            "interventions_per_training_seed": 7,
            "measurements_per_pair": int(matrix.shape[1]),
        },
        "definition": {
            "per_measurement_contrast": (
                "(abs(text rank change) - abs(image rank change)) / "
                "(abs(text rank change) + abs(image rank change)); zero when both are zero"
            ),
            "primary_continuous_score": "median of 21 normalized contrasts",
            "direction": "positive=text, negative=image, zero=tie",
        },
        "direction_counts": {
            str(key): int(value)
            for key, value in output["continuous_direction"].value_counts().items()
        },
        "input_sha256": input_hashes,
        "output_sha256": sha256_file(OUTPUT),
    }
    SUMMARY_OUTPUT.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
