from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dirty_blocking import run_dirty_pipeline


def normalize_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "id_left",
        "id_right",
        "record_left",
        "record_right",
        "label",
    }
    missing = required_columns - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidates is missing columns: {sorted(missing)}")

    df = candidates.copy()
    df["id_left"] = df["id_left"].astype(str)
    df["id_right"] = df["id_right"].astype(str)
    df["record_left"] = df["record_left"].astype(str)
    df["record_right"] = df["record_right"].astype(str)
    df["label"] = df["label"].astype(bool)

    inconsistent_left = (
        df.groupby("id_left")["record_left"].nunique(dropna=False).reset_index(name="n")
    )
    max_n = int(inconsistent_left["n"].max()) if len(inconsistent_left) else 0
    if max_n > 1:
        bad_ids = inconsistent_left[inconsistent_left["n"] > 1]["id_left"].tolist()[:10]
        raise ValueError(
            "Found id_left values with multiple record_left strings. "
            f"Examples: {bad_ids}"
        )

    return df.reset_index(drop=True)


def load_or_generate_candidates(
    dataset_name: str,
    reader_root: Path,
    candidates_csv: Path | None,
    topk: int,
    force_rebuild_index: bool,
    sample_frac: float | None,
    sample_n: int | None,
    sample_cluster_n: int | None,
    sample_seed: int,
) -> pd.DataFrame:
    if candidates_csv is not None:
        return pd.read_csv(candidates_csv)

    return run_dirty_pipeline(
        dataset_name=dataset_name,
        reader_root=reader_root,
        topk=topk,
        output_path=None,
        force_rebuild_index=force_rebuild_index,
        sample_frac=sample_frac,
        sample_n=sample_n,
        sample_cluster_n=sample_cluster_n,
        sample_seed=sample_seed,
    )
