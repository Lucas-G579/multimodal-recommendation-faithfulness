"""Prepare dry-run requests for the isolated 80-sample prompt development set."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llm_evaluation_protocol import protocol_hashes, render_request, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = PROJECT_ROOT / "data" / "manifests" / "llm_evaluation_samples.csv"
BLIND_INPUTS = PROJECT_ROOT / "data" / "manifests" / "llm_blind_inputs.jsonl"
OUTPUT = PROJECT_ROOT / "outputs" / "llm_prompt_development" / "requests.jsonl"
MANIFEST = PROJECT_ROOT / "data" / "manifests" / "llm_prompt_development.json"


def main() -> None:
    samples = pd.read_csv(SAMPLES)
    development = samples.loc[samples["cohort"] == "prompt_development"]
    if len(development) != 80:
        raise ValueError(f"Expected 80 development samples, found {len(development)}")
    allowed_ids = set(development["sample_id"])

    blind_records = []
    with BLIND_INPUTS.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["sample_id"] in allowed_ids:
                blind_records.append(record)
    if {record["sample_id"] for record in blind_records} != allowed_ids:
        raise ValueError("Blind inputs do not exactly cover development sample IDs")

    requests = [render_request(record) for record in blind_records]
    forbidden_fragments = (
        "cross_seed", "cohort", "contrast_median", "rank_change", "training_seed"
    )
    for request in requests:
        serialized = json.dumps(request, ensure_ascii=False).lower()
        if any(fragment in serialized for fragment in forbidden_fragments):
            raise ValueError(f"Answer-bearing content in {request['sample_id']}")
        if len(request["image_paths"]) < 1:
            raise ValueError(f"No target image in {request['sample_id']}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        for request in requests:
            handle.write(json.dumps(request, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "status": "DRY_RUN_PASSED",
        "protocol_version": "1.1-prompt-v1",
        "scope": "prompt_development_only",
        "requests": len(requests),
        "confirmatory_requests": 0,
        "answer_bearing_fragments_found": 0,
        "image_count": sum(len(request["image_paths"]) for request in requests),
        "protocol_hashes": protocol_hashes(),
        "inputs": {
            "samples_sha256": sha256_file(SAMPLES),
            "blind_inputs_sha256": sha256_file(BLIND_INPUTS),
        },
        "dry_run_output_sha256": sha256_file(OUTPUT),
    }
    MANIFEST.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
