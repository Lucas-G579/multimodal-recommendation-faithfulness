"""Run a single-epoch MMRec LightGCN smoke test on the Baby dataset."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


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


SMOKE_CONFIG = {
    "gpu_id": 0,
    "epochs": 1,
    "stopping_step": 1,
    "train_batch_size": 4096,
    "eval_batch_size": 4096,
    "metrics": ["Recall", "NDCG"],
    "topk": [10, 20],
    "valid_metric": "Recall@20",
    "save_recommended_topk": False,
    "seed": [2026],
    "n_layers": [1],
    "reg_weight": [1e-4],
    "hyper_parameters": ["seed", "n_layers", "reg_weight"],
}

if __name__ == "__main__":
    quick_start(
        model="LightGCN",
        dataset="baby",
        config_dict=SMOKE_CONFIG,
        save_model=False,
    )
