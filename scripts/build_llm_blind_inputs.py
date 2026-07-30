"""Build label-free, multimodal inputs for the frozen LLM evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = PROJECT_ROOT / "data" / "manifests" / "llm_evaluation_samples.csv"
METADATA = PROJECT_ROOT / "data" / "processed" / "baby_item_metadata.jsonl"
INTERACTIONS = PROJECT_ROOT / "external" / "MMRec" / "data" / "baby" / "baby.inter"
IMAGE_DIR = PROJECT_ROOT / "data" / "processed" / "llm_evaluation_images"
OUTPUT = PROJECT_ROOT / "data" / "manifests" / "llm_blind_inputs.jsonl"
SUMMARY = PROJECT_ROOT / "data" / "manifests" / "llm_blind_inputs.json"
HISTORY_LENGTH = 5
FORBIDDEN_TERMS = (
    "label",
    "cohort",
    "contrast",
    "rank_change",
    "direction_votes",
    "training_seed",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_metadata(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records[int(record["item_id"])] = record
    return records


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(normalize_text(part) for part in value).strip()
    return str(value).strip()


def image_entry(item_id: int) -> tuple[bool, str]:
    path = IMAGE_DIR / f"{item_id}.img"
    if not path.exists():
        return False, ""
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as error:
        raise ValueError(f"Invalid stored image for item_id={item_id}") from error
    return True, path.relative_to(PROJECT_ROOT).as_posix()


def item_entry(
    item_id: int,
    metadata: dict[int, dict[str, Any]],
    include_description: bool,
) -> dict[str, Any]:
    record = metadata[item_id]
    image_available, image_path = image_entry(item_id)
    entry = {
        "title": normalize_text(record.get("title")),
        "image_available": image_available,
        "image_path": image_path,
    }
    if include_description:
        entry["description"] = normalize_text(record.get("description"))
    return entry


def main() -> None:
    samples = pd.read_csv(SAMPLES)
    metadata = load_metadata(METADATA)
    interactions = pd.read_csv(INTERACTIONS, sep="\t")
    train = interactions.loc[interactions["x_label"] == 0].copy()
    train.sort_values(
        ["userID", "timestamp", "itemID"],
        ascending=[True, False, True],
        inplace=True,
    )
    histories = {
        int(user_id): [int(value) for value in group.head(HISTORY_LENGTH)["itemID"]]
        for user_id, group in train.groupby("userID", sort=False)
    }

    records: list[dict[str, Any]] = []
    history_image_total = 0
    history_image_available = 0
    for row in samples.itertuples(index=False):
        user_id = int(row.user_id)
        target_item_id = int(row.item_id)
        history_items = histories.get(user_id, [])
        history_entries = [
            item_entry(item_id, metadata, include_description=False)
            for item_id in history_items
        ]
        history_image_total += len(history_entries)
        history_image_available += sum(
            entry["image_available"] for entry in history_entries
        )
        records.append(
            {
                "sample_id": str(row.sample_id),
                "target": item_entry(
                    target_item_id, metadata, include_description=True
                ),
                "history": history_entries,
                "missingness": {
                    "target_title": not bool(
                        normalize_text(metadata[target_item_id].get("title"))
                    ),
                    "target_description": not bool(
                        normalize_text(metadata[target_item_id].get("description"))
                    ),
                    "target_image": not (IMAGE_DIR / f"{target_item_id}.img").exists(),
                    "history_images": len(history_entries)
                    - sum(entry["image_available"] for entry in history_entries),
                },
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )

    serialized_keys = set()
    for record in records:
        serialized_keys.update(record.keys())
        serialized_keys.update(record["target"].keys())
        for history_entry in record["history"]:
            serialized_keys.update(history_entry.keys())
    forbidden_found = sorted(
        key
        for key in serialized_keys
        if any(term in key.lower() for term in FORBIDDEN_TERMS)
    )
    if forbidden_found:
        raise ValueError(f"Forbidden answer-bearing fields: {forbidden_found}")

    target_images = sum(
        not record["missingness"]["target_image"] for record in records
    )
    target_titles = sum(
        not record["missingness"]["target_title"] for record in records
    )
    target_descriptions = sum(
        not record["missingness"]["target_description"] for record in records
    )
    summary = {
        "status": "FROZEN",
        "protocol_version": "1.1",
        "answer_bearing_fields_found": forbidden_found,
        "rows": len(records),
        "history": {
            "selection": (
                "up to 5 x_label=0 items per user; timestamp descending, "
                "item ID ascending tie-break"
            ),
            "entries": history_image_total,
            "image_coverage": float(
                history_image_available / history_image_total
            ),
        },
        "target_coverage": {
            "title": float(target_titles / len(records)),
            "description": float(target_descriptions / len(records)),
            "image": float(target_images / len(records)),
        },
        "inputs": {
            "samples_sha256": sha256_file(SAMPLES),
            "metadata_sha256": sha256_file(METADATA),
            "interactions_sha256": sha256_file(INTERACTIONS),
        },
        "output_sha256": sha256_file(OUTPUT),
    }
    SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
