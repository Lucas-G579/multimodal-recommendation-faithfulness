"""Validate MMRec image/text feature arrays against an interaction file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def inspect_array(path: Path, chunk_size: int = 512) -> dict[str, object]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 2:
        raise ValueError(f"{path} must be 2-D, got shape={array.shape}")

    finite = True
    zero_rows = 0
    norm_min = float("inf")
    norm_max = float("-inf")
    norm_sum = 0.0

    for start in range(0, array.shape[0], chunk_size):
        chunk = np.asarray(array[start : start + chunk_size])
        finite = finite and bool(np.isfinite(chunk).all())
        norms = np.linalg.norm(chunk, axis=1)
        zero_rows += int(np.count_nonzero(norms == 0))
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))
        norm_sum += float(norms.sum())

    return {
        "path": str(path.resolve()),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bytes": int(path.stat().st_size),
        "all_finite": finite,
        "zero_rows": zero_rows,
        "norm_min": norm_min,
        "norm_mean": norm_sum / array.shape[0],
        "norm_max": norm_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    args = parser.parse_args()

    interactions = pd.read_csv(args.interactions, sep="\t", usecols=["itemID"])
    unique_items = int(interactions["itemID"].nunique())
    max_item_id = int(interactions["itemID"].max())

    image = inspect_array(args.image)
    text = inspect_array(args.text)
    expected_rows = max_item_id + 1

    checks = {
        "interaction_unique_items": unique_items,
        "interaction_max_item_id": max_item_id,
        "expected_feature_rows": expected_rows,
        "image_rows_match": image["shape"][0] == expected_rows,
        "text_rows_match": text["shape"][0] == expected_rows,
        "modalities_row_aligned": image["shape"][0] == text["shape"][0],
    }

    result = {"checks": checks, "image": image, "text": text}
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if not all(
        [
            checks["image_rows_match"],
            checks["text_rows_match"],
            checks["modalities_row_aligned"],
            image["all_finite"],
            text["all_finite"],
        ]
    ):
        raise SystemExit("Feature validation failed")


if __name__ == "__main__":
    main()
