"""Verify that MGCN image/text branches causally affect inference scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import verify_mgcn_checkpoint as runtime


DEFAULT_OUTPUT = (
    runtime.runtime.PROJECT_ROOT
    / "outputs"
    / "audits"
    / "mgcn_modality_branch_probe.json"
)


@torch.no_grad()
def score(model: torch.nn.Module, users: torch.Tensor) -> torch.Tensor:
    return model.full_sort_predict([users]).detach().cpu()


def zero_output(
    _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
) -> torch.Tensor:
    return torch.zeros_like(output)


def summarize(
    baseline: torch.Tensor,
    intervened: torch.Tensor,
    seen_mask: torch.Tensor,
    topk: int,
) -> dict[str, Any]:
    absolute = (baseline - intervened).abs()
    baseline_rank = baseline.clone()
    intervened_rank = intervened.clone()
    baseline_rank[seen_mask] = -1e10
    intervened_rank[seen_mask] = -1e10
    baseline_topk = torch.topk(baseline_rank, topk, dim=1).indices
    intervened_topk = torch.topk(intervened_rank, topk, dim=1).indices
    overlap = [
        len(set(base.tolist()).intersection(changed.tolist())) / topk
        for base, changed in zip(baseline_topk, intervened_topk)
    ]
    return {
        "max_absolute_score_difference": float(absolute.max()),
        "mean_absolute_score_difference": float(absolute.mean()),
        "mean_topk_overlap": float(sum(overlap) / len(overlap)),
        "top1_change_rate": float(
            (baseline_topk[:, 0] != intervened_topk[:, 0]).float().mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=runtime.DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-users", type=int, default=32)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--numerical-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    model, _config, _valid_data, test_data = runtime.load_mgcn(
        args.checkpoint.resolve()
    )
    first_batch = next(iter(test_data))
    users = first_batch[0][: args.sample_users]
    seen_mask = torch.zeros(
        (args.sample_users, model.n_items), dtype=torch.bool
    )
    mask_rows, mask_items = first_batch[1][0].cpu(), first_batch[1][1].cpu()
    selected = mask_rows < args.sample_users
    seen_mask[mask_rows[selected], mask_items[selected]] = True

    baseline = score(model, users)
    unchanged_repeat = score(model, users)

    image_handle = model.gate_v.register_forward_hook(zero_output)
    image_zero = score(model, users)
    image_handle.remove()

    text_handle = model.gate_t.register_forward_hook(zero_output)
    text_zero = score(model, users)
    text_handle.remove()

    image_handle = model.gate_v.register_forward_hook(zero_output)
    text_handle = model.gate_t.register_forward_hook(zero_output)
    both_zero = score(model, users)
    image_handle.remove()
    text_handle.remove()

    null_summary = summarize(
        baseline, unchanged_repeat, seen_mask, args.topk
    )
    image_summary = summarize(baseline, image_zero, seen_mask, args.topk)
    text_summary = summarize(baseline, text_zero, seen_mask, args.topk)
    both_summary = summarize(baseline, both_zero, seen_mask, args.topk)
    passed = (
        null_summary["max_absolute_score_difference"] <= args.numerical_tolerance
        and image_summary["max_absolute_score_difference"] > args.numerical_tolerance
        and text_summary["max_absolute_score_difference"] > args.numerical_tolerance
        and both_summary["max_absolute_score_difference"] > args.numerical_tolerance
    )

    result = {
        "status": "PASSED" if passed else "FAILED",
        "checkpoint_sha256": runtime.runtime.sha256_file(
            args.checkpoint.resolve()
        ),
        "intervention": (
            "Forward hooks set the output of MGCN's modality-specific "
            "behavior-guided gate to zero. Parameters are not modified."
        ),
        "sample_users": args.sample_users,
        "scores_per_condition": int(baseline.numel()),
        "topk": args.topk,
        "numerical_tolerance": args.numerical_tolerance,
        "unchanged_repeat_control": null_summary,
        "image_branch_zero": image_summary,
        "text_branch_zero": text_summary,
        "both_modality_branches_zero": both_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit("MGCN modality branch probe failed")


if __name__ == "__main__":
    main()
