import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from llm_evaluation_protocol import ResponseValidationError, parse_response


def valid_response() -> dict:
    return {
        "explanation": "The visible text and image both match the user's history.",
        "primary_evidence": "both",
        "claimed_image_share": 0.4,
        "claimed_text_share": 0.6,
        "confidence": 0.7,
        "conflict_detected": False,
        "abstained": False,
    }


class ResponseProtocolTests(unittest.TestCase):
    def test_accepts_valid_response(self) -> None:
        expected = valid_response()
        self.assertEqual(parse_response(json.dumps(expected)), expected)

    def test_accepts_consistent_abstention(self) -> None:
        value = valid_response()
        value.update(
            primary_evidence="insufficient",
            claimed_image_share=0,
            claimed_text_share=0,
            abstained=True,
        )
        self.assertEqual(parse_response(json.dumps(value)), value)

    def test_rejects_markdown_fence(self) -> None:
        with self.assertRaises(ResponseValidationError):
            parse_response("```json\n" + json.dumps(valid_response()) + "\n```")

    def test_rejects_extra_key(self) -> None:
        value = valid_response()
        value["hidden_label"] = "text"
        with self.assertRaises(ResponseValidationError):
            parse_response(json.dumps(value))

    def test_rejects_shares_not_summing_to_one(self) -> None:
        value = valid_response()
        value["claimed_text_share"] = 0.5
        with self.assertRaises(ResponseValidationError):
            parse_response(json.dumps(value))

    def test_rejects_inconsistent_abstention(self) -> None:
        value = valid_response()
        value["abstained"] = True
        with self.assertRaises(ResponseValidationError):
            parse_response(json.dumps(value))

    def test_rejects_boolean_as_number(self) -> None:
        value = valid_response()
        value["confidence"] = True
        with self.assertRaises(ResponseValidationError):
            parse_response(json.dumps(value))


if __name__ == "__main__":
    unittest.main()
