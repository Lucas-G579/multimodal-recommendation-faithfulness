"""Reload the accepted BM3 checkpoint and verify its evaluation results.

This is the reference (no-intervention) path for later modality audits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MMREC_SRC = PROJECT_ROOT / "external" / "MMRec" / "src"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "checkpoints" / "mmrec" / "BM3-baby-best.pth"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "audits" / "bm3_checkpoint_verification.json"

os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / "cache" / "matplotlib")
)
MMREC_SRC.mkdir(parents=True, exist_ok=True)
os.chdir(MMREC_SRC)
sys.path.insert(0, str(MMREC_SRC))

# Compatibility shim for MMRec's evaluator under NumPy >= 1.24.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

from common.trainer import Trainer  # noqa: E402
from utils.configurator import Config  # noqa: E402
from utils.dataloader import EvalDataLoader, TrainDataLoader  # noqa: E402
from utils.dataset import RecDataset  # noqa: E402
from utils.utils import get_model, init_seed  # noqa: E402


CONFIG_OVERRIDES: dict[str, Any] = {
    "gpu_id": 0,
    "use_gpu": True,
    "seed": 999,
    "train_batch_size": 2048,
    "eval_batch_size": 4096,
    "metrics": ["Recall", "NDCG", "Precision", "MAP"],
    "topk": [5, 10, 20, 50],
    "valid_metric": "Recall@20",
    "n_layers": 1,
    "dropout": 0.5,
    "reg_weight": 0.1,
    "hyper_parameters": [],
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def compare_metrics(
    expected: dict[str, float], actual: dict[str, float], tolerance: float
) -> dict[str, dict[str, float | bool]]:
    comparison = {}
    for name, expected_value in expected.items():
        actual_value = float(actual[name])
        difference = abs(float(expected_value) - actual_value)
        comparison[name] = {
            "expected": float(expected_value),
            "actual": actual_value,
            "absolute_difference": difference,
            "passed": difference <= tolerance,
        }
    return comparison


@torch.no_grad()
def make_topk_fingerprint(
    model: torch.nn.Module,
    eval_data: EvalDataLoader,
    sample_users: int,
    topk: int,
) -> dict[str, Any]:
    """Create a deterministic fingerprint from the first evaluation users."""
    model.eval()
    selected_users: list[int] = []
    selected_items: list[list[int]] = []

    for batched_data in eval_data:
        scores = model.full_sort_predict(batched_data)
        masked_items = batched_data[1]
        scores[masked_items[0], masked_items[1]] = -1e10
        topk_items = torch.topk(scores, topk, dim=-1).indices.cpu()

        remaining = sample_users - len(selected_users)
        if remaining <= 0:
            break
        take = min(remaining, topk_items.shape[0])
        batch_users = batched_data[0][:take].detach().cpu().tolist()
        selected_users.extend(int(user) for user in batch_users)
        selected_items.extend(topk_items[:take].tolist())

        if len(selected_users) == sample_users:
            break

    payload = json.dumps(
        {"users": selected_users, "topk_items": selected_items},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "sample_users": len(selected_users),
        "topk": topk,
        "sha256": hashlib.sha256(payload).hexdigest().upper(),
        "users": selected_users,
        "topk_items": selected_items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metric-tolerance", type=float, default=1e-12)
    parser.add_argument("--sample-users", type=int, default=32)
    parser.add_argument("--fingerprint-topk", type=int, default=20)
    args = parser.parse_args()

    if args.sample_users <= 0:
        raise ValueError("--sample-users must be positive")
    if args.fingerprint_topk <= 0:
        raise ValueError("--fingerprint-topk must be positive")

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    required_keys = {
        "dataset",
        "epoch",
        "model",
        "state_dict",
        "test_result",
        "valid_result",
    }
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        raise KeyError(f"Checkpoint missing keys: {sorted(missing_keys)}")
    if checkpoint["model"] != "BM3" or checkpoint["dataset"] != "baby":
        raise ValueError(
            f"Expected BM3/baby, got {checkpoint['model']}/{checkpoint['dataset']}"
        )

    config = Config("BM3", "baby", dict(CONFIG_OVERRIDES))
    init_seed(config["seed"])
    dataset = RecDataset(config)
    train_dataset, valid_dataset, test_dataset = dataset.split()
    # MMRec initializes ``inter_num`` as a side effect of dataset.__str__().
    # quick_start logs every split before constructing loaders; reproduce that
    # otherwise-hidden initialization step in this standalone loader.
    for split_dataset in (train_dataset, valid_dataset, test_dataset):
        str(split_dataset)
    train_data = TrainDataLoader(
        config,
        train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=False,
    )
    valid_data = EvalDataLoader(
        config,
        valid_dataset,
        additional_dataset=train_dataset,
        batch_size=config["eval_batch_size"],
    )
    test_data = EvalDataLoader(
        config,
        test_dataset,
        additional_dataset=train_dataset,
        batch_size=config["eval_batch_size"],
    )

    model = get_model("BM3")(config, train_data).to(config["device"])
    load_result = model.load_state_dict(checkpoint["state_dict"], strict=True)
    trainer = Trainer(config, model)

    valid_result = trainer.evaluate(valid_data)
    test_result = trainer.evaluate(test_data)
    valid_comparison = compare_metrics(
        checkpoint["valid_result"], valid_result, args.metric_tolerance
    )
    test_comparison = compare_metrics(
        checkpoint["test_result"], test_result, args.metric_tolerance
    )
    fingerprint = make_topk_fingerprint(
        model, test_data, args.sample_users, args.fingerprint_topk
    )

    all_metrics_passed = all(
        item["passed"]
        for comparison in (valid_comparison, test_comparison)
        for item in comparison.values()
    )
    result = {
        "status": "PASSED" if all_metrics_passed else "FAILED",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": int(checkpoint["epoch"]),
            "model": checkpoint["model"],
            "dataset": checkpoint["dataset"],
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(config["device"]),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "load_state_dict": {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        },
        "metric_tolerance": args.metric_tolerance,
        "valid_comparison": valid_comparison,
        "test_comparison": test_comparison,
        "test_topk_fingerprint": fingerprint,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if not all_metrics_passed:
        raise SystemExit("Checkpoint metric verification failed")


if __name__ == "__main__":
    main()
