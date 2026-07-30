"""Recover and verify MMRec Baby item-ID mappings from Amazon 2014 data."""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RATINGS = PROJECT_ROOT / "data" / "raw" / "amazon_baby_2014" / "ratings_Baby.csv"
DEFAULT_METADATA = (
    PROJECT_ROOT / "data" / "raw" / "amazon_baby_2014" / "meta_Baby.json.gz"
)
CURRENT_INTERACTIONS = (
    PROJECT_ROOT / "external" / "MMRec" / "data" / "baby" / "baby.inter"
)
DEFAULT_MAPPING = PROJECT_ROOT / "data" / "processed" / "baby_item_mapping.tsv"
DEFAULT_ITEMS = PROJECT_ROOT / "data" / "processed" / "baby_item_metadata.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "baby_metadata_recovery.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def filter_five_core(frame: pd.DataFrame) -> None:
    while True:
        user_counts = Counter(frame["userID"].to_numpy())
        item_counts = Counter(frame["asin"].to_numpy())
        bad_users = {key for key, value in user_counts.items() if value < 5}
        bad_items = {key for key, value in item_counts.items() if value < 5}
        if not bad_users and not bad_items:
            return
        drop = frame["userID"].isin(bad_users) | frame["asin"].isin(bad_items)
        frame.drop(frame.index[drop], inplace=True)


def rebuild_interactions(ratings_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(
        ratings_path,
        names=["userID", "asin", "rating", "timestamp"],
        header=None,
        dtype={"userID": str, "asin": str},
    )
    ratings.dropna(subset=["userID", "asin", "timestamp"], inplace=True)
    ratings.drop_duplicates(
        subset=["userID", "asin", "timestamp"], inplace=True
    )
    filter_five_core(ratings)
    ratings.reset_index(drop=True, inplace=True)

    user_map = {
        raw_id: index for index, raw_id in enumerate(pd.unique(ratings["userID"]))
    }
    item_map = {
        asin: index for index, asin in enumerate(pd.unique(ratings["asin"]))
    }
    mapping = pd.DataFrame(
        [(asin, item_id) for asin, item_id in item_map.items()],
        columns=["asin", "itemID"],
    )
    ratings["userID"] = ratings["userID"].map(user_map).astype(int)
    ratings["itemID"] = ratings["asin"].map(item_map).astype(int)

    rebuilt = ratings[
        ["userID", "itemID", "rating", "timestamp"]
    ]
    return rebuilt, mapping


def metadata_records(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = ast.literal_eval(line)
            except (SyntaxError, ValueError) as error:
                raise ValueError(
                    f"Invalid metadata record at line {line_number}"
                ) from error
            if isinstance(value, dict):
                yield value


def clean_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--mapping-output", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--items-output", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    rebuilt, mapping = rebuild_interactions(args.ratings)
    current = pd.read_csv(CURRENT_INTERACTIONS, sep="\t")
    interaction_columns = ["userID", "itemID", "rating", "timestamp"]
    rebuilt_counts = (
        rebuilt[interaction_columns].value_counts().sort_index()
    )
    current_counts = (
        current[interaction_columns].value_counts().sort_index()
    )
    if not rebuilt_counts.equals(current_counts):
        raise ValueError(
            "Rebuilt interaction multiset does not exactly match current MMRec data"
        )

    asin_to_item = dict(zip(mapping["asin"], mapping["itemID"]))
    selected: dict[int, dict[str, Any]] = {}
    for record in metadata_records(args.metadata):
        asin = str(record.get("asin", ""))
        item_id = asin_to_item.get(asin)
        if item_id is None or item_id in selected:
            continue
        selected[item_id] = {
            "item_id": int(item_id),
            "asin": asin,
            "title": clean_value(record.get("title")),
            "description": clean_value(record.get("description")),
            "brand": clean_value(record.get("brand")),
            "price": clean_value(record.get("price")),
            "categories": clean_value(record.get("categories")),
            "image_url": clean_value(record.get("imUrl")),
        }

    args.mapping_output.parent.mkdir(parents=True, exist_ok=True)
    args.items_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(args.mapping_output, sep="\t", index=False)
    with args.items_output.open("w", encoding="utf-8", newline="\n") as handle:
        for item_id in sorted(selected):
            handle.write(
                json.dumps(selected[item_id], ensure_ascii=False, sort_keys=True)
                + "\n"
            )

    total_items = len(mapping)
    coverage = {
        "metadata": len(selected),
        "title": sum(bool(value.get("title")) for value in selected.values()),
        "description": sum(
            bool(value.get("description")) for value in selected.values()
        ),
        "image_url": sum(
            bool(value.get("image_url")) for value in selected.values()
        ),
    }
    manifest = {
        "status": "PASSED",
        "source": {
            "dataset": "Amazon product data 2014 / Baby",
            "ratings_sha256": sha256_file(args.ratings),
            "metadata_sha256": sha256_file(args.metadata),
        },
        "verification": {
            "interaction_multiset_exact_match": True,
            "split_note": (
                "MMRec preprocessing shuffles without a recorded seed before "
                "assigning per-user train/valid/test labels; mapping is created "
                "before that shuffle, so the full interaction multiset is the "
                "recoverable identity check"
            ),
            "split_counts": {
                str(key): int(value)
                for key, value in current["x_label"].value_counts().sort_index().items()
            },
            "rows": int(len(rebuilt)),
            "users": int(rebuilt["userID"].nunique()),
            "items": total_items,
        },
        "coverage": {
            key: {
                "count": int(value),
                "fraction": float(value / total_items),
            }
            for key, value in coverage.items()
        },
        "outputs": {
            "mapping_path": str(args.mapping_output),
            "mapping_sha256": sha256_file(args.mapping_output),
            "items_path": str(args.items_output),
            "items_sha256": sha256_file(args.items_output),
        },
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
