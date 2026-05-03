from __future__ import annotations

import math
from typing import Any

import networkx as nx
import pandas as pd


def truncate_text(text: str, max_chars: int = 240) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 3]}..."


def canonical_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))


def build_entity_record_lookup(predicted_df: pd.DataFrame) -> dict[str, str]:
    record_lookup: dict[str, str] = {}
    for left, right, record_left, record_right in predicted_df[
        ["id_left", "id_right", "record_left", "record_right"]
    ].itertuples(index=False, name=None):
        record_lookup[str(left)] = str(record_left)
        record_lookup[str(right)] = str(record_right)
    return record_lookup


def build_pair_prediction_lookup(predicted_df: pd.DataFrame) -> dict[tuple[str, str], bool]:
    lookup: dict[tuple[str, str], bool] = {}
    for left, right, pred in predicted_df[["id_left", "id_right", "pred"]].itertuples(
        index=False,
        name=None,
    ):
        key = canonical_pair(str(left), str(right))
        lookup[key] = bool(pred)
    return lookup


def build_entity_cluster_lookup(predicted_df: pd.DataFrame) -> dict[str, str]:
    left_col = "cluster_id_left"
    right_col = "cluster_id_right"
    if left_col not in predicted_df.columns or right_col not in predicted_df.columns:
        return {}

    cluster_lookup: dict[str, str] = {}
    for left, right, left_cluster, right_cluster in predicted_df[
        ["id_left", "id_right", left_col, right_col]
    ].itertuples(index=False, name=None):
        cluster_lookup[str(left)] = str(left_cluster)
        cluster_lookup[str(right)] = str(right_cluster)
    return cluster_lookup


def gt_partition_for_triplet(
    a: str,
    b: str,
    c: str,
    cluster_lookup: dict[str, str],
) -> dict[str, str]:
    ca = cluster_lookup.get(a)
    cb = cluster_lookup.get(b)
    cc = cluster_lookup.get(c)

    if ca is None or cb is None or cc is None:
        partition = "unknown"
    elif ca == cb == cc:
        partition = "{A,B,C}"
    elif ca == cb and ca != cc:
        partition = "{A,B}|{C}"
    elif ca == cc and ca != cb:
        partition = "{A,C}|{B}"
    elif cb == cc and ca != cb:
        partition = "{B,C}|{A}"
    else:
        partition = "{A}|{B}|{C}"

    return {
        "gt_cluster_a": "" if ca is None else ca,
        "gt_cluster_b": "" if cb is None else cb,
        "gt_cluster_c": "" if cc is None else cc,
        "gt_partition": partition,
    }


def build_edge_outcome_lookup(predicted_df: pd.DataFrame) -> dict[tuple[str, str], str]:
    outcome_lookup: dict[tuple[str, str], str] = {}
    for left, right, pred, label in predicted_df[
        ["id_left", "id_right", "pred", "label"]
    ].itertuples(index=False, name=None):
        key = canonical_pair(str(left), str(right))
        pred_bool = bool(pred)
        label_bool = bool(label)
        if pred_bool and label_bool:
            outcome = "TP"
        elif pred_bool and not label_bool:
            outcome = "FP"
        elif (not pred_bool) and label_bool:
            outcome = "FN"
        else:
            outcome = "TN"
        outcome_lookup[key] = outcome
    return outcome_lookup


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    q = min(1.0, max(0.0, float(q)))
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = q * (len(ordered) - 1)
    low = int(math.floor(idx))
    high = int(math.ceil(idx))
    if low == high:
        return float(ordered[low])
    frac = idx - low
    return float((1.0 - frac) * ordered[low] + frac * ordered[high])


def cluster_internal_layout(
    nodes: list[str], subgraph: nx.Graph
) -> dict[str, tuple[float, float]]:
    if len(nodes) == 1:
        return {nodes[0]: (0.0, 0.0)}

    if subgraph.number_of_edges() == 0:
        angles = [2.0 * math.pi * idx / len(nodes) for idx in range(len(nodes))]
        return {
            node: (math.cos(angle), math.sin(angle))
            for node, angle in zip(nodes, angles, strict=True)
        }

    layout = nx.spring_layout(subgraph, seed=42)
    return {str(node): (float(x), float(y)) for node, (x, y) in layout.items()}
