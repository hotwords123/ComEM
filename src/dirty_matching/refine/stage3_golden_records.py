from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from tqdm.contrib.concurrent import thread_map

from src.dirty_matching.core.lookups import build_entity_record_lookup
from src.dirty_matching.llm.local_resolvers import GoldenRecordBuilder
from src.dirty_matching.refine.cache_paths import dirty_matching_stage_cache_dir
from src.dirty_matching.visualization.golden_graph import (
    visualize_golden_graph_html,
    visualize_golden_graph_png,
)


def _select_one_medoid(members: list[str], subgraph: nx.Graph) -> tuple[str, float]:
    if not members:
        return "", 0.0
    if len(members) == 1:
        return str(members[0]), 1.0

    # Use a finite penalty for unreachable nodes so disconnected subgraphs remain comparable.
    penalty = len(members) + 1
    best_node = ""
    best_distance_sum = float("inf")
    best_degree = -1.0

    for node in members:
        node_str = str(node)
        lengths = nx.single_source_shortest_path_length(subgraph, node_str)
        distance_sum = 0.0
        for other in members:
            other_str = str(other)
            if other_str == node_str:
                continue
            distance_sum += float(lengths.get(other_str, penalty))

        degree_score = float(subgraph.degree(node_str, weight="weight"))
        if (
            distance_sum < best_distance_sum
            or (
                distance_sum == best_distance_sum
                and (
                    degree_score > best_degree
                    or (
                        degree_score == best_degree
                        and (best_node == "" or node_str < best_node)
                    )
                )
            )
        ):
            best_node = node_str
            best_distance_sum = distance_sum
            best_degree = degree_score

    max_total = float((len(members) - 1) * penalty)
    confidence = 1.0 - (best_distance_sum / max_total) if max_total > 0 else 1.0
    confidence = min(1.0, max(0.0, confidence))
    return best_node, confidence


def stage3_build_golden_records(
    predicted_df: pd.DataFrame,
    local_clusters: dict[str, list[str]],
    local_cluster_to_community: dict[str, int],
    out_dir: Path,
    model_name: str,
    max_workers: int,
    max_retries: int,
    golden_method: str = "medoid",
) -> dict[str, Any]:
    stage_dir = out_dir / "stage3_golden_records"
    stage_dir.mkdir(parents=True, exist_ok=True)

    method = str(golden_method).strip().lower()
    if method not in {"medoid", "llm"}:
        raise ValueError(f"Unsupported golden method: {golden_method}")

    record_lookup = build_entity_record_lookup(predicted_df)
    builder = (
        GoldenRecordBuilder(
            model_name=model_name,
            max_retries=max_retries,
            cache_dir=dirty_matching_stage_cache_dir(model_name, "stage3"),
        )
        if method == "llm"
        else None
    )

    local_items = sorted(local_clusters.items(), key=lambda item: item[0])

    def run_llm_build(item: tuple[str, list[str]]) -> dict[str, Any]:
        local_cluster_id, members = item
        assert builder is not None
        return builder.build(
            local_cluster_id=local_cluster_id,
            entity_ids=members,
            record_lookup=record_lookup,
        )

    result_lookup: dict[str, dict[str, Any]] = {}
    if method == "llm":
        build_results = (
            thread_map(run_llm_build, local_items, max_workers=max(1, int(max_workers)))
            if local_items
            else []
        )
        result_lookup = {
            str(result["local_cluster_id"]): result for result in build_results
        }
    else:
        positive_graph = nx.Graph()
        for left, right, pred in predicted_df[["id_left", "id_right", "pred"]].itertuples(
            index=False,
            name=None,
        ):
            if not bool(pred):
                continue
            positive_graph.add_edge(str(left), str(right), weight=1)

        for local_cluster_id, members in local_items:
            member_ids = sorted(str(member) for member in members)
            subgraph = positive_graph.subgraph(member_ids).copy()
            subgraph.add_nodes_from(member_ids)
            medoid_id, confidence = _select_one_medoid(member_ids, subgraph)
            result_lookup[local_cluster_id] = {
                "local_cluster_id": local_cluster_id,
                "golden_record": str(record_lookup.get(medoid_id, "")),
                "confidence": float(confidence),
                "status": "graph_1_medoid",
                "attempts": 0,
                "selected_entity_id": medoid_id,
            }

    golden_rows: list[dict[str, Any]] = []
    entity_rows: list[dict[str, Any]] = []
    golden_records: dict[str, str] = {}
    entity_to_golden: dict[str, str] = {}
    golden_to_community: dict[str, int] = {}
    golden_to_member_size: dict[str, int] = {}

    for idx, (local_cluster_id, members) in enumerate(local_items):
        result = result_lookup.get(local_cluster_id)
        if result is None:
            result = {
                "local_cluster_id": local_cluster_id,
                "golden_record": " | ".join(record_lookup.get(m, "") for m in members[:3]),
                "confidence": 0.1,
                "status": "missing_result_fallback",
                "attempts": 0,
            }

        golden_id = f"g{idx}"
        golden_record = str(result.get("golden_record", ""))
        golden_records[golden_id] = golden_record
        golden_to_community[golden_id] = int(
            local_cluster_to_community.get(local_cluster_id, -1)
        )
        golden_to_member_size[golden_id] = int(len(members))
        golden_rows.append(
            {
                "golden_id": golden_id,
                "local_cluster_id": local_cluster_id,
                "community_id": golden_to_community[golden_id],
                "member_size": int(len(members)),
                "member_ids": json.dumps(sorted(members), ensure_ascii=True),
                "golden_record": golden_record,
                "confidence": float(result.get("confidence", 0.5)),
                "status": str(result.get("status", "")),
                "attempts": int(result.get("attempts", 0)),
                "selected_entity_id": str(result.get("selected_entity_id", "")),
                "method": method,
            }
        )
        for entity_id in members:
            entity_to_golden[entity_id] = golden_id
            entity_rows.append(
                {
                    "entity_id": entity_id,
                    "local_cluster_id": local_cluster_id,
                    "golden_id": golden_id,
                }
            )

    golden_csv = stage_dir / "golden_records.csv"
    pd.DataFrame(golden_rows).to_csv(golden_csv, index=False)
    entity_csv = stage_dir / "entity_golden_mapping.csv"
    pd.DataFrame(entity_rows).to_csv(entity_csv, index=False)

    golden_graph = nx.Graph()
    for golden_id in sorted(golden_records.keys()):
        golden_graph.add_node(
            golden_id,
            community_id=int(golden_to_community.get(golden_id, -1)),
            member_size=int(golden_to_member_size.get(golden_id, 1)),
            record_preview=str(golden_records.get(golden_id, ""))[:180],
        )

    for left, right, pred in predicted_df[["id_left", "id_right", "pred"]].itertuples(
        index=False,
        name=None,
    ):
        if not bool(pred):
            continue
        left_id = str(left)
        right_id = str(right)
        golden_left = entity_to_golden.get(left_id)
        golden_right = entity_to_golden.get(right_id)
        if golden_left is None or golden_right is None or golden_left == golden_right:
            continue

        if golden_graph.has_edge(golden_left, golden_right):
            golden_graph[golden_left][golden_right]["weight"] += 1
            support_pairs = golden_graph[golden_left][golden_right]["support_pairs"]
            if len(support_pairs) < 4:
                support_pairs.append([left_id, right_id])
        else:
            golden_graph.add_edge(
                golden_left,
                golden_right,
                weight=1,
                support_pairs=[[left_id, right_id]],
            )

    golden_graph_png = stage_dir / "golden_graph.png"
    golden_graph_html = stage_dir / "golden_graph.html"
    title = "Stage3 Golden-Record Graph"
    visualize_golden_graph_png(golden_graph, golden_graph_png, title=title)
    visualize_golden_graph_html(golden_graph, golden_graph_html, title=title)

    status_counts = (
        pd.Series([row["status"] for row in golden_rows]).value_counts().to_dict()
        if golden_rows
        else {}
    )
    stats = {
        "golden_records": int(len(golden_rows)),
        "entities": int(len(entity_to_golden)),
        "golden_graph_nodes": int(golden_graph.number_of_nodes()),
        "golden_graph_edges": int(golden_graph.number_of_edges()),
        "status_counts": status_counts,
        "method": method,
        "api_cost": float(builder.cost) if builder is not None else 0.0,
    }
    summary_json = stage_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=True, indent=2)

    return {
        "golden_records": golden_records,
        "entity_to_golden": entity_to_golden,
        "stats": stats,
        "files": {
            "golden_records_csv": str(golden_csv),
            "entity_mapping_csv": str(entity_csv),
            "golden_graph_png": str(golden_graph_png),
            "golden_graph_html": str(golden_graph_html),
            "summary_json": str(summary_json),
        },
    }
