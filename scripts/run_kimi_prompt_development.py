"""Run only the frozen 80-sample prompt-development set through Kimi."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_evaluation_protocol import (
    ResponseValidationError,
    parse_response,
    protocol_hashes,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUESTS = PROJECT_ROOT / "outputs" / "llm_prompt_development" / "requests.jsonl"
SAMPLES = PROJECT_ROOT / "data" / "manifests" / "llm_evaluation_samples.csv"
OUTPUT = PROJECT_ROOT / "outputs" / "llm_prompt_development" / "kimi_responses.jsonl"
SUMMARY = PROJECT_ROOT / "outputs" / "llm_prompt_development" / "kimi_summary.json"
ENDPOINT = "https://api.moonshot.cn/v1/chat/completions"
MODEL = "kimi-k2.6"
MAX_COMPLETION_TOKENS = 512
PROTOCOL_VERSION = "v1-natural-explanation"
EXPECTED_REQUESTS = 80
MAX_SAMPLE_LIMIT = 80
EXPERIMENT_SCOPE = "prompt_development"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def data_url(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0]
    if media_type is None or not media_type.startswith("image/"):
        # Downloaded files deliberately use .img; let Pillow-validated bytes use JPEG.
        media_type = "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{payload}"


def request_payload(request: dict[str, Any]) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for relative_path in request["image_paths"]:
        path = PROJECT_ROOT / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Missing request image: {relative_path}")
        content.append(
            {"type": "image_url", "image_url": {"url": data_url(path)}}
        )
    content.append({"type": "text", "text": request["user_prompt"]})
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": request["system_prompt"]},
            {"role": "user", "content": content},
        ],
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "stream": False,
    }


def post_json(payload: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT,
        data=encoded,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "FaithRec-MM-academic-audit/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def latest_by_sample(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        latest[record["sample_id"]] = record
    return latest


def attempt_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        sample_id = record["sample_id"]
        counts[sample_id] = max(counts.get(sample_id, 0), int(record["attempt"]))
    return counts


def append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-attempts-per-sample", type=int, default=1)
    parser.add_argument("--continue-after-failure", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= MAX_SAMPLE_LIMIT:
        raise ValueError(f"--limit must be between 1 and {MAX_SAMPLE_LIMIT}")
    if not 1 <= args.max_attempts_per_sample <= 2:
        raise ValueError("--max-attempts-per-sample must be 1 or 2")
    api_key = os.environ.get("MOONSHOT_API_KEY", "")
    if not api_key:
        raise RuntimeError("MOONSHOT_API_KEY is not present in this process")

    with REQUESTS.open(encoding="utf-8") as handle:
        requests = [json.loads(line) for line in handle]
    if len(requests) != EXPECTED_REQUESTS:
        raise ValueError(
            f"Expected exactly {EXPECTED_REQUESTS} requests, got {len(requests)}"
        )

    records_before = load_records(OUTPUT)
    existing = latest_by_sample(records_before)
    counts_before = attempt_counts(records_before)
    pending = [
        request for request in requests
        if request["sample_id"] not in existing
        or (
            existing[request["sample_id"]]["status"] != "valid"
            and counts_before.get(request["sample_id"], 0)
            < args.max_attempts_per_sample
        )
    ][: args.limit]
    if not pending:
        print("No pending development requests within the requested limit.")
        return

    for index, request in enumerate(pending, start=1):
        payload = request_payload(request)
        payload_bytes = json.dumps(
            payload, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
        prior_attempts = counts_before.get(request["sample_id"], 0)
        record: dict[str, Any] | None = None
        for attempt in range(prior_attempts + 1, args.max_attempts_per_sample + 1):
            if attempt > prior_attempts + 1:
                time.sleep(2)
            started = time.monotonic()
            record = {
                "sample_id": request["sample_id"],
                "protocol_version": PROTOCOL_VERSION,
                "experiment_scope": EXPERIMENT_SCOPE,
                "status": "api_error",
                "attempt": attempt,
                "requested_model": MODEL,
                "returned_model": "",
                "endpoint": ENDPOINT,
                "thinking": "disabled",
                "temperature": "unsupported_by_provider",
                "seed": "unsupported_by_provider",
                "max_completion_tokens": MAX_COMPLETION_TOKENS,
                "request_sha256": hashlib.sha256(payload_bytes).hexdigest().upper(),
                "request_image_count": len(request["image_paths"]),
                "started_at_utc": utc_now(),
                "finished_at_utc": "",
                "latency_seconds": 0.0,
                "finish_reason": "",
                "usage": {},
                "raw_content": "",
                "parsed": None,
                "error_type": "",
                "error_message": "",
                "response_id": "",
            }
            try:
                response = post_json(payload, api_key, args.timeout)
                message = response["choices"][0]["message"]
                raw_content = message.get("content") or ""
                record.update(
                    {
                        "returned_model": response.get("model", ""),
                        "finish_reason": response["choices"][0].get("finish_reason", ""),
                        "usage": response.get("usage", {}),
                        "raw_content": raw_content,
                        "response_id": response.get("id", ""),
                    }
                )
                try:
                    record["parsed"] = parse_response(raw_content)
                    record["status"] = "valid"
                except ResponseValidationError as error:
                    record["status"] = "schema_invalid"
                    record["error_type"] = type(error).__name__
                    record["error_message"] = str(error)
            except urllib.error.HTTPError as error:
                body = error.read().decode("utf-8", errors="replace")
                record["error_type"] = f"HTTPError_{error.code}"
                record["error_message"] = body[:1000]
            except Exception as error:
                record["error_type"] = type(error).__name__
                record["error_message"] = str(error)[:1000]
            finally:
                record["finished_at_utc"] = utc_now()
                record["latency_seconds"] = round(time.monotonic() - started, 6)
                append_record(OUTPUT, record)
            print(
                f"sample={index}/{len(pending)} attempt={attempt}/"
                f"{args.max_attempts_per_sample} sample_id={record['sample_id']} "
                f"status={record['status']}",
                flush=True,
            )
            if record["status"] == "valid":
                break
        assert record is not None
        if record["status"] != "valid" and not args.continue_after_failure:
            raise SystemExit(2)

    all_attempts = load_records(OUTPUT)
    all_records = latest_by_sample(all_attempts)
    statuses: dict[str, int] = {}
    for record in all_records.values():
        statuses[record["status"]] = statuses.get(record["status"], 0) + 1
    usage_fields = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens")
    summary = {
        "status": "DEVELOPMENT_IN_PROGRESS",
        "model": MODEL,
        "protocol_version": PROTOCOL_VERSION,
        "experiment_scope": EXPERIMENT_SCOPE,
        "records": len(all_records),
        "attempt_records": len(all_attempts),
        "max_attempts_per_sample": args.max_attempts_per_sample,
        "status_counts": statuses,
        "usage_totals": {
            field: sum(int(record.get("usage", {}).get(field, 0) or 0) for record in all_records.values())
            for field in usage_fields
        },
        "protocol_hashes": protocol_hashes(),
        "requests_sha256": sha256_file(REQUESTS),
        "responses_sha256": sha256_file(OUTPUT),
    }
    SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
