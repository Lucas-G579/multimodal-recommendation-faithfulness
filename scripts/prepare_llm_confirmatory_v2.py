"""Prepare the frozen, label-free 590-sample confirmatory request set."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llm_evaluation_protocol_v2 import protocol_hashes, render_request, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = PROJECT_ROOT / "data" / "manifests" / "llm_evaluation_samples.csv"
BLIND_INPUTS = PROJECT_ROOT / "data" / "manifests" / "llm_blind_inputs.jsonl"
OUTPUT = PROJECT_ROOT / "outputs" / "llm_confirmatory_v2" / "requests.jsonl"
MANIFEST = PROJECT_ROOT / "data" / "manifests" / "llm_confirmatory_v2.json"
FORBIDDEN = (
    "cross_seed", "cohort", "contrast", "rank_change", "direction_votes",
    "training_seed", "strict_a", "a_or_b",
)


def collect_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key).lower())
            keys.update(collect_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(collect_keys(child))
    return keys


def main() -> None:
    samples = pd.read_csv(SAMPLES)
    allowed = set(
        samples.loc[samples["cohort"] == "primary_confirmatory", "sample_id"]
    )
    development = set(
        samples.loc[samples["cohort"] == "prompt_development", "sample_id"]
    )
    if len(allowed) != 590:
        raise ValueError(f"Expected 590 confirmatory samples, found {len(allowed)}")
    if allowed & development:
        raise ValueError("Development and confirmatory sample IDs overlap")

    records = []
    with BLIND_INPUTS.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["sample_id"] in allowed:
                records.append(record)
    if {record["sample_id"] for record in records} != allowed:
        raise ValueError("Blind inputs do not exactly cover confirmatory IDs")
    requests = [render_request(record) for record in records]
    for request in requests:
        keys = collect_keys(request)
        found = [
            fragment for fragment in FORBIDDEN
            if any(fragment in key for key in keys)
        ]
        if found:
            raise ValueError(
                f"Answer-bearing content in {request['sample_id']}: {found}"
            )
        if len(request["image_paths"]) < 1:
            raise ValueError(f"No target image in {request['sample_id']}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        for request in requests:
            handle.write(json.dumps(request, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "status": "FROZEN_NOT_RUN",
        "protocol_version": "v2-forced-choice",
        "scope": "primary_confirmatory_only",
        "requests": len(requests),
        "development_requests": 0,
        "answer_bearing_fragments_found": 0,
        "image_count": sum(len(request["image_paths"]) for request in requests),
        "protocol_hashes": protocol_hashes(),
        "inputs": {
            "samples_sha256": sha256_file(SAMPLES),
            "blind_inputs_sha256": sha256_file(BLIND_INPUTS),
        },
        "requests_sha256": sha256_file(OUTPUT),
    }
    MANIFEST.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
