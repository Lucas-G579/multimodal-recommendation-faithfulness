import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from llm_evaluation_protocol_v2 import ResponseValidationError, parse_response


def response(evidence="text", image_share=0.3, text_share=0.7, abstained=False):
    return {
        "explanation": "The text provides the stronger match.",
        "primary_evidence": evidence,
        "claimed_image_share": image_share,
        "claimed_text_share": text_share,
        "confidence": 0.7,
        "conflict_detected": False,
        "abstained": abstained,
    }


class ForcedChoiceProtocolTests(unittest.TestCase):
    def test_accepts_consistent_text(self):
        value = response()
        self.assertEqual(parse_response(json.dumps(value)), value)

    def test_accepts_consistent_image(self):
        value = response("image", 0.8, 0.2)
        self.assertEqual(parse_response(json.dumps(value)), value)

    def test_accepts_insufficient(self):
        value = response("insufficient", 0, 0, True)
        self.assertEqual(parse_response(json.dumps(value)), value)

    def test_rejects_both(self):
        value = response("both", 0.5, 0.5)
        with self.assertRaises(ResponseValidationError):
            parse_response(json.dumps(value))

    def test_rejects_tied_shares(self):
        with self.assertRaises(ResponseValidationError):
            parse_response(json.dumps(response("text", 0.5, 0.5)))

    def test_rejects_label_share_contradiction(self):
        with self.assertRaises(ResponseValidationError):
            parse_response(json.dumps(response("image", 0.2, 0.8)))


if __name__ == "__main__":
    unittest.main()
