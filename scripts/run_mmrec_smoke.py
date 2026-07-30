"""Run controlled MMRec LightGCN/BM3 checks on the Baby dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MMREC_SRC = PROJECT_ROOT / "external" / "MMRec" / "src"
MMREC_DATA = PROJECT_ROOT / "external" / "MMRec" / "data" / "baby" / "baby.inter"
MPL_CACHE = PROJECT_ROOT / "outputs" / "cache" / "matplotlib"

if not MMREC_SRC.exists():
    raise FileNotFoundError(f"MMRec source not found: {MMREC_SRC}")
if not MMREC_DATA.exists():
    raise FileNotFoundError(f"Baby interactions not found: {MMREC_DATA}")

# MMRec resolves configs and ../data relative to the current working directory.
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CACHE)

# MMRec's evaluator uses the removed NumPy alias ``np.float`` in four places.
# Keep the downloaded upstream snapshot unchanged and provide a local shim.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

os.chdir(MMREC_SRC)
sys.path.insert(0, str(MMREC_SRC))

from utils.quick_start import quick_start  # noqa: E402


BASE_CONFIG = {
    "gpu_id": 0,
    "checkpoint_dir": str(PROJECT_ROOT / "outputs" / "checkpoints" / "mmrec"),
    "epochs": 1,
    "stopping_step": 1,
    "train_batch_size": 4096,
    "eval_batch_size": 4096,
    "metrics": ["Recall", "NDCG"],
    "topk": [10, 20],
    "valid_metric": "Recall@20",
    "save_recommended_topk": False,
    "seed": [2026],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["LightGCN", "BM3", "MGCN"], default="LightGCN"
    )
    parser.add_argument(
        "--profile", choices=["smoke", "official"], default="smoke"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Training seed. Defaults to 2026 for smoke and 999 for official.",
    )
    args = parser.parse_args()
    selected_seed = args.seed
    if selected_seed is None:
        selected_seed = 999 if args.profile == "official" else 2026

    model_config = {
        "LightGCN": {
            "n_layers": [1],
            "reg_weight": [1e-4],
            "hyper_parameters": ["seed", "n_layers", "reg_weight"],
        },
        "BM3": {
            "n_layers": [1],
            "dropout": [0.3],
            "reg_weight": [0.01],
            "hyper_parameters": ["seed", "n_layers", "dropout", "reg_weight"],
        },
        "MGCN": {
            "cl_loss": [0.01],
            "hyper_parameters": ["seed", "cl_loss"],
        },
    }

    run_config = BASE_CONFIG | model_config[args.model] | {
        "seed": [selected_seed]
    }
    save_model = False
    checkpoint_path = None
    started_at = datetime.now(timezone.utc)
    if args.profile == "official":
        checkpoint_dir = (
            PROJECT_ROOT
            / "outputs"
            / "checkpoints"
            / "mmrec"
            / f"{args.model.lower()}_baby_seed_{selected_seed}"
        )
        checkpoint_path = checkpoint_dir / f"{args.model}-baby-best.pth"
        if checkpoint_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing checkpoint: {checkpoint_path}"
            )
        common_official = {
            "epochs": 1000,
            "stopping_step": 20,
            "train_batch_size": 2048,
            "eval_batch_size": 4096,
            "metrics": ["Recall", "NDCG", "Precision", "MAP"],
            "topk": [5, 10, 20, 50],
            "seed": [selected_seed],
            "checkpoint_dir": str(checkpoint_dir),
        }
        if args.model == "BM3":
            run_config |= common_official | {
                "n_layers": [1],
                "dropout": [0.5],
                "reg_weight": [0.1],
            }
        elif args.model == "MGCN":
            run_config |= common_official | {
                "cl_loss": [0.01],
                "knn_k": 20,
            }
        else:
            raise ValueError(
                "The official profile is currently defined for BM3 and MGCN"
            )
        save_model = True

    quick_start(
        model=args.model,
        dataset="baby",
        config_dict=run_config,
        save_model=save_model,
    )

    if checkpoint_path is not None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Training finished without expected checkpoint: {checkpoint_path}"
            )
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        manifest = {
            "status": "COMPLETED",
            "model": args.model,
            "dataset": "baby",
            "seed": selected_seed,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": digest.hexdigest().upper(),
                "epoch": int(checkpoint["epoch"]),
            },
            "valid_result": checkpoint["valid_result"],
            "test_result": checkpoint["test_result"],
            "configuration": {
                "epochs": 1000,
                "stopping_step": 20,
                "train_batch_size": 2048,
                "eval_batch_size": 4096,
                "valid_metric": "Recall@20",
                "cl_loss": 0.01 if args.model == "MGCN" else None,
                "knn_k": 20 if args.model == "MGCN" else None,
            },
        }
        manifest_path = checkpoint_path.with_name("run_manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
