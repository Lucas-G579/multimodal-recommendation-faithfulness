"""Print the minimum environment facts needed for reproducible experiments."""

from __future__ import annotations

import json
import platform
import subprocess
import sys


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


facts: dict[str, object] = {
    "python": sys.version,
    "platform": platform.platform(),
    "executable": sys.executable,
    "git": command_output(["git", "--version"]),
    "nvidia_smi": command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free",
            "--format=csv,noheader",
        ]
    ),
}

try:
    import torch

    facts["torch"] = {
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
except ImportError:
    facts["torch"] = None

print(json.dumps(facts, ensure_ascii=False, indent=2))
