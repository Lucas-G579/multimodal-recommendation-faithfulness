"""Run the frozen 590-sample v2 confirmatory set without exposing labels."""

from pathlib import Path

import llm_evaluation_protocol_v2 as protocol_v2
import run_kimi_prompt_development as runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
runner.REQUESTS = PROJECT_ROOT / "outputs" / "llm_confirmatory_v2" / "requests.jsonl"
runner.OUTPUT = PROJECT_ROOT / "outputs" / "llm_confirmatory_v2" / "kimi_responses.jsonl"
runner.SUMMARY = PROJECT_ROOT / "outputs" / "llm_confirmatory_v2" / "kimi_summary.json"
runner.PROTOCOL_VERSION = "v2-forced-choice"
runner.EXPECTED_REQUESTS = 590
runner.MAX_SAMPLE_LIMIT = 590
runner.EXPERIMENT_SCOPE = "primary_confirmatory"
runner.parse_response = protocol_v2.parse_response
runner.protocol_hashes = protocol_v2.protocol_hashes
runner.ResponseValidationError = protocol_v2.ResponseValidationError


if __name__ == "__main__":
    runner.main()
