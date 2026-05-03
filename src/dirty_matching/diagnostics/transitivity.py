from __future__ import annotations

from itertools import combinations
from typing import Any

import networkx as nx
import pandas as pd

from src.dirty_matching.core.lookups import (
    build_entity_cluster_lookup,
    build_pair_prediction_lookup,
    gt_partition_for_triplet,
)


def collect_mode_violations(
    graph: nx.Graph,
    pair_lookup: dict[tuple[str, str], bool],
    cluster_lookup: dict[str, str],
    mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if mode not in {"strict_negative", "include_missing"}:
        raise ValueError("mode must be one of: strict_negative, include_missing")

    violations: list[dict[str, Any]] = []
    total_wedges = 0

    for center in graph.nodes():
        neighbors = sorted(str(n) for n in graph.neighbors(center))
        if len(neighbors) < 2:
            continue

        for a, c in combinations(neighbors, 2):
            total_wedges += 1
            key_ac = tuple(sorted((a, c)))
            has_ac = key_ac in pair_lookup
            ac_pred = pair_lookup.get(key_ac)
            ac_is_match = bool(ac_pred) if has_ac else False

            violated = False
            if mode == "strict_negative":
                violated = has_ac and (ac_pred is False)
            elif mode == "include_missing":
                violated = not ac_is_match

            if violated:
                gt_info = gt_partition_for_triplet(a, str(center), c, cluster_lookup)
                violations.append(
                    {
                        "a": a,
                        "b": str(center),
                        "c": c,
                        "mode": mode,
                        "has_ac": bool(has_ac),
                        "ac_pred": None if not has_ac else bool(ac_pred),
                        "gt_cluster_a": gt_info["gt_cluster_a"],
                        "gt_cluster_b": gt_info["gt_cluster_b"],
                        "gt_cluster_c": gt_info["gt_cluster_c"],
                        "gt_partition": gt_info["gt_partition"],
                    }
                )

    involved_nodes = sorted(
        {node for row in violations for node in (row["a"], row["b"], row["c"])}
    )
    violating_pairs = sorted({tuple(sorted((row["a"], row["c"]))) for row in violations})
    gt_partition_counts = (
        pd.DataFrame(violations)["gt_partition"].value_counts().to_dict()
        if violations
        else {}
    )
    stats = {
        "mode": mode,
        "total_wedges": int(total_wedges),
        "violating_triplets": int(len(violations)),
        "violating_ac_pairs": int(len(violating_pairs)),
        "involved_nodes": int(len(involved_nodes)),
        "violation_rate": float(len(violations) / total_wedges) if total_wedges else 0.0,
        "gt_partition_counts": gt_partition_counts,
    }
    return violations, stats


def detect_transitivity_violations(
    graph: nx.Graph,
    predicted_df: pd.DataFrame,
    mode: str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, Any]]]:
    pair_lookup = build_pair_prediction_lookup(predicted_df)
    cluster_lookup = build_entity_cluster_lookup(predicted_df)

    requested_violations, requested_stats = collect_mode_violations(
        graph,
        pair_lookup,
        cluster_lookup,
        mode,
    )
    strict_violations, strict_stats = collect_mode_violations(
        graph,
        pair_lookup,
        cluster_lookup,
        "strict_negative",
    )
    include_violations, include_stats = collect_mode_violations(
        graph,
        pair_lookup,
        cluster_lookup,
        "include_missing",
    )

    comparison = {
        "strict_negative": strict_stats,
        "include_missing": include_stats,
        "delta_include_minus_strict": int(len(include_violations) - len(strict_violations)),
    }

    violations_df = pd.DataFrame(requested_violations)
    if len(violations_df) == 0:
        violations_df = pd.DataFrame(
            columns=[
                "a",
                "b",
                "c",
                "mode",
                "has_ac",
                "ac_pred",
                "gt_cluster_a",
                "gt_cluster_b",
                "gt_cluster_c",
                "gt_partition",
            ]
        )

    return violations_df, requested_stats, comparison
