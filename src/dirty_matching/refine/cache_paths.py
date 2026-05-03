from __future__ import annotations

from pathlib import Path


def dirty_matching_stage_cache_dir(model_name: str, stage: str) -> Path:
    return Path("results/diskcache") / f"dirty_matching_{model_name}" / stage
