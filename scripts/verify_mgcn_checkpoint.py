"""Reload and strictly verify the accepted MGCN/Baby checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import verify_bm3_checkpoint as runtime


DEFAULT_CHECKPOINT = (
    runtime.PROJECT_ROOT / "outputs" / "checkpoints" / "mmrec" / "MGCN-baby-best.pth"
)
DEFAULT_OUTPUT = (
    runtime.PROJECT_ROOT / "outputs" / "audits" / "mgcn_checkpoint_verification.json"
)
CONFIG_OVERRIDES = runtime.CONFIG_OVERRIDES | {
    "cl_loss": 0.01,
    "knn_k": 20,
}


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (runtime.PROJECT_ROOT / path).resolve()


def infer_training_seed(checkpoint_path: Path) -> int:
    manifest_path = checkpoint_path.with_name("run_manifest.json")
    if not manifest_path.is_file():
        return 999
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model") != "MGCN" or manifest.get("dataset") != "baby":
        raise ValueError(f"Unexpected run manifest identity: {manifest_path}")
    return int(manifest["seed"])


def load_mgcn(
    checkpoint_path: Path,
    training_seed: int | None = None,
) -> tuple[torch.nn.Module, runtime.Config, runtime.EvalDataLoader, runtime.EvalDataLoader]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint["model"] != "MGCN" or checkpoint["dataset"] != "baby":
        raise ValueError(
            f"Expected MGCN/baby, got {checkpoint['model']}/{checkpoint['dataset']}"
        )

    if training_seed is None:
        training_seed = infer_training_seed(checkpoint_path)
    config_overrides = dict(CONFIG_OVERRIDES)
    config_overrides["seed"] = training_seed
    config = runtime.Config("MGCN", "baby", config_overrides)
    runtime.init_seed(config["seed"])
    dataset = runtime.RecDataset(config)
    train_dataset, valid_dataset, test_dataset = dataset.split()
    for split_dataset in (train_dataset, valid_dataset, test_dataset):
        str(split_dataset)

    train_data = runtime.TrainDataLoader(
        config,
        train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=False,
    )
    valid_data = runtime.EvalDataLoader(
        config,
        valid_dataset,
        additional_dataset=train_dataset,
        batch_size=config["eval_batch_size"],
    )
    test_data = runtime.EvalDataLoader(
        config,
        test_dataset,
        additional_dataset=train_dataset,
        batch_size=config["eval_batch_size"],
    )
    model = runtime.get_model("MGCN")(config, train_data).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, config, valid_data, test_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metric-tolerance", type=float, default=1e-12)
    parser.add_argument("--sample-users", type=int, default=32)
    parser.add_argument(
        "--training-seed",
        type=int,
        help="Override the seed inferred from a sibling run_manifest.json.",
    )
    args = parser.parse_args()

    checkpoint_path = project_path(args.checkpoint)
    args.output = project_path(args.output)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    training_seed = (
        args.training_seed
        if args.training_seed is not None
        else infer_training_seed(checkpoint_path)
    )
    model, config, valid_data, test_data = load_mgcn(
        checkpoint_path, training_seed=training_seed
    )
    trainer = runtime.Trainer(config, model)

    valid_result = trainer.evaluate(valid_data)
    test_result = trainer.evaluate(test_data)
    valid_comparison = runtime.compare_metrics(
        checkpoint["valid_result"], valid_result, args.metric_tolerance
    )
    test_comparison = runtime.compare_metrics(
        checkpoint["test_result"], test_result, args.metric_tolerance
    )
    fingerprint = runtime.make_topk_fingerprint(
        model, test_data, args.sample_users, topk=20
    )
    passed = all(
        item["passed"]
        for comparison in (valid_comparison, test_comparison)
        for item in comparison.values()
    )

    result = {
        "status": "PASSED" if passed else "FAILED",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": runtime.sha256_file(checkpoint_path),
            "epoch": int(checkpoint["epoch"]),
        },
        "configuration": {
            "cl_loss": 0.01,
            "knn_k": 20,
            "seed": training_seed,
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
    if not passed:
        raise SystemExit("MGCN checkpoint verification failed")


if __name__ == "__main__":
    main()
