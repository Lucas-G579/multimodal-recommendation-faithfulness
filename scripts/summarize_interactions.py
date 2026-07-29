"""Summarize an MMRec interaction file without modifying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("path", type=Path)
args = parser.parse_args()

frame = pd.read_csv(args.path, sep="\t")
required = {"userID", "itemID", "x_label"}
missing = required.difference(frame.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

summary = {
    "path": str(args.path.resolve()),
    "rows": int(len(frame)),
    "users": int(frame["userID"].nunique()),
    "items": int(frame["itemID"].nunique()),
    "duplicate_user_item_rows": int(frame.duplicated(["userID", "itemID"]).sum()),
    "missing_values": int(frame.isna().sum().sum()),
    "split_counts": {
        str(key): int(value)
        for key, value in frame["x_label"].value_counts().sort_index().items()
    },
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
