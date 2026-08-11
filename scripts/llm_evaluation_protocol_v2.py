"""Forced-choice v2 prompt rendering and strict response validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "llm_faithfulness_v2_system.txt"
USER_PROMPT_PATH = PROJECT_ROOT / "prompts" / "llm_faithfulness_v2_user.txt"
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "llm_faithfulness_response_v2.json"
EVIDENCE_VALUES = {"image", "text", "insufficient"}
REQUIRED_KEYS = {
    "explanation", "primary_evidence", "claimed_image_share",
    "claimed_text_share", "confidence", "conflict_detected", "abstained",
}


class ResponseValidationError(ValueError):
    """Raised when a response violates the frozen v2 contract."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def protocol_hashes() -> dict[str, str]:
    return {
        "system_prompt_sha256": sha256_file(SYSTEM_PROMPT_PATH),
        "user_template_sha256": sha256_file(USER_PROMPT_PATH),
        "response_schema_sha256": sha256_file(SCHEMA_PATH),
    }


def render_request(record: dict[str, Any]) -> dict[str, Any]:
    target = record["target"]
    history_lines: list[str] = []
    image_paths = [target["image_path"]]
    for index, entry in enumerate(record["history"], start=1):
        availability = "supplied" if entry["image_available"] else "missing"
        history_lines.append(
            f"{index}. Title: {entry['title'] or '[missing]'}; image: {availability}."
        )
        if entry["image_available"]:
            image_paths.append(entry["image_path"])
    history_text = "\n".join(history_lines) or "No visible history is available."
    return {
        "sample_id": record["sample_id"],
        "system_prompt": SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
        "user_prompt": USER_PROMPT_PATH.read_text(encoding="utf-8").format(
            target_title=target["title"] or "[missing]",
            target_description=target.get("description") or "[missing]",
            history_text=history_text,
        ),
        "image_paths": image_paths,
    }


def parse_response(raw: str) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        raise ResponseValidationError("response is empty or not a string")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ResponseValidationError(f"invalid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise ResponseValidationError("top-level response must be an object")
    if set(value) != REQUIRED_KEYS:
        raise ResponseValidationError(
            f"key mismatch; missing={sorted(REQUIRED_KEYS-set(value))}, "
            f"extra={sorted(set(value)-REQUIRED_KEYS)}"
        )
    if not isinstance(value["explanation"], str) or not value["explanation"].strip():
        raise ResponseValidationError("explanation must be a non-empty string")
    if len(value["explanation"]) > 1200:
        raise ResponseValidationError("explanation exceeds 1200 characters")
    evidence = value["primary_evidence"]
    if evidence not in EVIDENCE_VALUES:
        raise ResponseValidationError("invalid primary_evidence")
    for key in ("claimed_image_share", "claimed_text_share", "confidence"):
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise ResponseValidationError(f"{key} must be numeric")
        if not 0 <= float(number) <= 1:
            raise ResponseValidationError(f"{key} must be in [0, 1]")
    for key in ("conflict_detected", "abstained"):
        if not isinstance(value[key], bool):
            raise ResponseValidationError(f"{key} must be boolean")
    image_share = float(value["claimed_image_share"])
    text_share = float(value["claimed_text_share"])
    if evidence == "insufficient":
        if not value["abstained"] or image_share != 0 or text_share != 0:
            raise ResponseValidationError(
                "insufficient requires abstained=true and both shares=0"
            )
    else:
        if value["abstained"]:
            raise ResponseValidationError("abstained=true requires insufficient")
        if abs(image_share + text_share - 1.0) > 1e-6:
            raise ResponseValidationError("non-abstaining shares must sum to 1")
        if image_share == text_share:
            raise ResponseValidationError("non-abstaining shares must not tie")
        if evidence == "image" and image_share <= text_share:
            raise ResponseValidationError("image label contradicts shares")
        if evidence == "text" and text_share <= image_share:
            raise ResponseValidationError("text label contradicts shares")
    return value
