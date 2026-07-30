"""Download and validate images for the frozen LLM evaluation samples."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = PROJECT_ROOT / "data" / "manifests" / "llm_evaluation_samples.csv"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "baby_item_metadata.jsonl"
DEFAULT_INTERACTIONS = (
    PROJECT_ROOT / "external" / "MMRec" / "data" / "baby" / "baby.inter"
)
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "processed" / "llm_evaluation_images"
DEFAULT_RECORDS = (
    PROJECT_ROOT / "data" / "manifests" / "llm_evaluation_image_download.csv"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT / "data" / "manifests" / "llm_evaluation_image_download.json"
)
USER_AGENT = "FaithRec-MM academic reproducibility audit/1.0"


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


def validate_image(payload: bytes) -> tuple[str, int, int]:
    with Image.open(io.BytesIO(payload)) as image:
        image.verify()
    with Image.open(io.BytesIO(payload)) as image:
        image_format = str(image.format or "UNKNOWN")
        width, height = image.size
    if width < 2 or height < 2:
        raise ValueError(f"implausible image dimensions {width}x{height}")
    return image_format, int(width), int(height)


def download_one(
    item_id: int,
    source_url: str,
    image_dir: Path,
    timeout: float,
) -> dict[str, Any]:
    normalized_url = (
        "https://" + source_url[len("http://") :]
        if source_url.startswith("http://")
        else source_url
    )
    output_path = image_dir / f"{item_id}.img"
    result: dict[str, Any] = {
        "item_id": item_id,
        "source_url": source_url,
        "request_url": normalized_url,
        "status": "failed",
        "attempts": 0,
        "http_status": "",
        "error_type": "",
        "error_message": "",
        "bytes": 0,
        "format": "",
        "width": "",
        "height": "",
        "sha256": "",
        "local_path": str(output_path),
    }

    if output_path.exists():
        try:
            payload = output_path.read_bytes()
            image_format, width, height = validate_image(payload)
            result.update(
                {
                    "status": "reused_valid",
                    "attempts": "",
                    "bytes": len(payload),
                    "format": image_format,
                    "width": width,
                    "height": height,
                    "sha256": hashlib.sha256(payload).hexdigest().upper(),
                }
            )
            return result
        except Exception:
            output_path.unlink()

    for attempt in (1, 2):
        result["attempts"] = attempt
        try:
            request = urllib.request.Request(
                normalized_url,
                headers={"User-Agent": USER_AGENT, "Accept": "image/*"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result["http_status"] = int(response.status)
                payload = response.read()
            image_format, width, height = validate_image(payload)
            output_path.write_bytes(payload)
            result.update(
                {
                    "status": "downloaded",
                    "attempts": "",
                    "bytes": len(payload),
                    "format": image_format,
                    "width": width,
                    "height": height,
                    "sha256": hashlib.sha256(payload).hexdigest().upper(),
                    "error_type": "",
                    "error_message": "",
                }
            )
            return result
        except urllib.error.HTTPError as error:
            result["http_status"] = int(error.code)
            result["error_type"] = type(error).__name__
            result["error_message"] = str(error)[:300]
        except Exception as error:
            result["error_type"] = type(error).__name__
            result["error_message"] = str(error)[:300]
        if attempt == 1:
            time.sleep(0.25)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--interactions", type=Path, default=DEFAULT_INTERACTIONS)
    parser.add_argument("--history-length", type=int, default=5)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    samples = pd.read_csv(args.samples)
    metadata = load_metadata(args.metadata)
    interactions = pd.read_csv(args.interactions, sep="\t")
    train = interactions.loc[
        (interactions["x_label"] == 0)
        & (interactions["userID"].isin(set(samples["user_id"])))
    ].copy()
    train.sort_values(
        ["userID", "timestamp", "itemID"],
        ascending=[True, False, True],
        inplace=True,
    )
    history = train.groupby("userID", sort=False).head(args.history_length)
    selected_items = sorted(
        {int(value) for value in samples["item_id"]}
        | {int(value) for value in history["itemID"]}
    )
    missing_metadata = [item_id for item_id in selected_items if item_id not in metadata]
    if missing_metadata:
        raise ValueError(f"Missing metadata for item IDs: {missing_metadata[:10]}")

    tasks_with_urls = [
        (item_id, str(metadata[item_id].get("image_url") or ""))
        for item_id in selected_items
    ]
    missing_urls = [item_id for item_id, url in tasks_with_urls if not url]
    tasks = [(item_id, url) for item_id, url in tasks_with_urls if url]

    args.image_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = [
        {
            "item_id": item_id,
            "source_url": "",
            "request_url": "",
            "status": "missing_url",
            "attempts": 0,
            "http_status": "",
            "error_type": "MissingImageURL",
            "error_message": "metadata image_url is empty",
            "bytes": 0,
            "format": "",
            "width": "",
            "height": "",
            "sha256": "",
            "local_path": str(args.image_dir / f"{item_id}.img"),
        }
        for item_id in missing_urls
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_one, item_id, url, args.image_dir, args.timeout
            ): item_id
            for item_id, url in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 100 == 0 or completed == len(futures):
                print(f"completed={completed}/{len(futures)}", flush=True)

    frame = pd.DataFrame(results).sort_values("item_id")
    args.records.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.records, index=False)
    frame.loc[
        frame["status"].isin(["downloaded", "reused_valid"]), "status"
    ] = "valid"
    successful = frame["status"] == "valid"
    successful_items = set(frame.loc[successful, "item_id"])
    sample_successes = samples["item_id"].isin(successful_items)
    summary = {
        "status": "PASSED" if float(sample_successes.mean()) >= 0.90 else "FAILED",
        "protocol": {
            "history_selection": (
                f"up to {args.history_length} most recent x_label=0 interactions; "
                "timestamp descending then item ID ascending"
            ),
            "source_url_policy": "original URL with http upgraded to https",
            "attempts_per_item": 2,
            "validation": "Pillow decode verification and dimensions >= 2x2",
            "silent_replacement": False,
        },
        "items": {
            "requested": len(selected_items),
            "successful": int(successful.sum()),
            "coverage": float(successful.mean()),
            "status_counts": {
                str(key): int(value)
                for key, value in frame["status"].value_counts().items()
            },
        },
        "samples": {
            "requested": int(len(samples)),
            "successful": int(sample_successes.sum()),
            "coverage": float(sample_successes.mean()),
            "threshold": 0.90,
        },
        "inputs": {
            "samples_sha256": sha256_file(args.samples),
            "metadata_sha256": sha256_file(args.metadata),
            "interactions_sha256": sha256_file(args.interactions),
        },
        "records_sha256": sha256_file(args.records),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASSED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
