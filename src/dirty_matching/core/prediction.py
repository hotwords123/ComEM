from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.contrib.concurrent import thread_map

from src.dirty_matching.core.candidates import normalize_candidates
from src.matching import Matching


def predict_candidates(
    candidates: pd.DataFrame,
    model_name: str,
    max_workers: int = 16,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = normalize_candidates(candidates)
    matcher = Matching(model_name=model_name)

    grouped = list(df.groupby("id_left", sort=False))
    instances: list[dict[str, Any]] = []
    for _, group in grouped:
        instances.append(
            {
                "anchor": group["record_left"].iloc[0],
                "candidates": group["record_right"].tolist(),
                "row_indexes": group.index.tolist(),
            }
        )

    def run_instance(instance: dict[str, Any]) -> tuple[list[int], list[bool]]:
        preds = matcher(instance)
        return instance["row_indexes"], preds

    mapped = thread_map(run_instance, instances, max_workers=max_workers)
    pred_series = pd.Series(index=df.index, dtype="boolean")
    for row_indexes, preds in mapped:
        if len(row_indexes) != len(preds):
            raise RuntimeError("Prediction length mismatch within one grouped instance")
        pred_series.loc[row_indexes] = preds

    if pred_series.isna().any():
        missing_indexes = pred_series[pred_series.isna()].index.tolist()[:10]
        raise RuntimeError(f"Missing predictions for row indexes: {missing_indexes}")

    df["pred"] = pred_series.astype(bool)

    labels = df["label"].tolist()
    preds = df["pred"].tolist()

    report = classification_report(
        labels, preds, digits=4, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(labels, preds, labels=[False, True]).tolist()

    metrics = {
        "model_name": model_name,
        "num_pairs": int(len(df)),
        "positive_labels": int(sum(labels)),
        "positive_predictions": int(sum(preds)),
        "classification_report": report,
        "confusion_matrix": {
            "labels": [False, True],
            "matrix": matrix,
        },
        "api_cost": float(matcher.cost),
    }
    return df, metrics
