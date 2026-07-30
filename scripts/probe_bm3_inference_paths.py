"""Empirically test which BM3 parameters affect inference-time scores.

Image/text features supervise BM3 during training, but ``full_sort_predict`` may
not consume them directly. This probe prevents invalid post-hoc modality masking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import verify_bm3_checkpoint as runtime


DEFAULT_OUTPUT = (
    runtime.PROJECT_ROOT / "outputs" / "audits" / "bm3_inference_path_probe.json"
)


def build_model(checkpoint_path: Path) -> tuple[torch.nn.Module, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = runtime.Config("BM3", "baby", dict(runtime.CONFIG_OVERRIDES))
    runtime.init_seed(config["seed"])
    dataset = runtime.RecDataset(config)
    train_dataset, _, _ = dataset.split()
    str(train_dataset)
    train_data = runtime.TrainDataLoader(
        config,
        train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=False,
    )
    model = runtime.get_model("BM3")(config, train_data).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, config


@torch.no_grad()
def score_users(model: torch.nn.Module, users: torch.Tensor) -> torch.Tensor:
    return model.full_sort_predict([users]).detach().cpu()


def difference_summary(
    baseline: torch.Tensor, intervened: torch.Tensor, numerical_tolerance: float
) -> dict[str, float | bool]:
    absolute = (baseline - intervened).abs()
    return {
        "max_absolute_difference": float(absolute.max()),
        "mean_absolute_difference": float(absolute.mean()),
        "exceeds_numerical_tolerance": bool(
            torch.any(absolute > numerical_tolerance)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=runtime.DEFAULT_CHECKPOINT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-users", type=int, default=32)
    parser.add_argument("--numerical-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    model, config = build_model(args.checkpoint.resolve())
    users = torch.arange(args.sample_users, device=config["device"])
    baseline = score_users(model, users)
    unchanged_repeat = score_users(model, users)

    image_original = model.image_embedding.weight.detach().clone()
    image_projection_original = {
        key: value.detach().clone() for key, value in model.image_trs.state_dict().items()
    }
    with torch.no_grad():
        model.image_embedding.weight.zero_()
        for parameter in model.image_trs.parameters():
            parameter.zero_()
    image_zero = score_users(model, users)
    with torch.no_grad():
        model.image_embedding.weight.copy_(image_original)
        model.image_trs.load_state_dict(image_projection_original)

    text_original = model.text_embedding.weight.detach().clone()
    text_projection_original = {
        key: value.detach().clone() for key, value in model.text_trs.state_dict().items()
    }
    with torch.no_grad():
        model.text_embedding.weight.zero_()
        for parameter in model.text_trs.parameters():
            parameter.zero_()
    text_zero = score_users(model, users)
    with torch.no_grad():
        model.text_embedding.weight.copy_(text_original)
        model.text_trs.load_state_dict(text_projection_original)

    # Positive control: a parameter that full_sort_predict actually consumes.
    item_original = model.item_id_embedding.weight.detach().clone()
    with torch.no_grad():
        model.item_id_embedding.weight.zero_()
    item_id_zero = score_users(model, users)
    with torch.no_grad():
        model.item_id_embedding.weight.copy_(item_original)

    null_result = difference_summary(
        baseline, unchanged_repeat, args.numerical_tolerance
    )
    image_result = difference_summary(
        baseline, image_zero, args.numerical_tolerance
    )
    text_result = difference_summary(
        baseline, text_zero, args.numerical_tolerance
    )
    positive_control = difference_summary(
        baseline, item_id_zero, args.numerical_tolerance
    )
    passed = (
        not null_result["exceeds_numerical_tolerance"]
        and not image_result["exceeds_numerical_tolerance"]
        and not text_result["exceeds_numerical_tolerance"]
        and positive_control["exceeds_numerical_tolerance"]
    )

    result = {
        "status": "PASSED" if passed else "FAILED",
        "interpretation": (
            "BM3 image/text tensors do not directly affect full_sort_predict; "
            "post-checkpoint zero masking is not a valid modality-dependence test."
        ),
        "checkpoint_sha256": runtime.sha256_file(args.checkpoint.resolve()),
        "sample_users": args.sample_users,
        "score_count_per_condition": int(baseline.numel()),
        "numerical_tolerance": args.numerical_tolerance,
        "unchanged_repeat_control": null_result,
        "image_zero_intervention": image_result,
        "text_zero_intervention": text_result,
        "item_id_zero_positive_control": positive_control,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if not passed:
        raise SystemExit("BM3 inference-path probe failed")


if __name__ == "__main__":
    main()
