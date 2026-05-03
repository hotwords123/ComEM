from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.contrib.concurrent import thread_map

from src.dirty_matching.core.lookups import canonical_pair
from src.dirty_matching.llm.global_resolver import GlobalConflictResolver
from src.dirty_matching.refine.cache_paths import dirty_matching_stage_cache_dir
from src.dirty_matching.refine.union_find import UnionFind


def collect_macro_edge_evidence(
    predicted_df: pd.DataFrame,
    entity_to_golden: dict[str, str],
) -> dict[tuple[str, str], dict[str, Any]]:
    evidence: dict[tuple[str, str], dict[str, Any]] = {}

    for left, right, pred in predicted_df[["id_left", "id_right", "pred"]].itertuples(
        index=False,
        name=None,
    ):
        left_id = str(left)
        right_id = str(right)
        golden_left = entity_to_golden.get(left_id)
        golden_right = entity_to_golden.get(right_id)
        if golden_left is None or golden_right is None or golden_left == golden_right:
            continue

        key = canonical_pair(golden_left, golden_right)
        bucket = evidence.setdefault(
            key,
            {
                "eq_count": 0,
                "neq_count": 0,
                "support_pairs_eq": [],
                "support_pairs_neq": [],
            },
        )
        if bool(pred):
            bucket["eq_count"] += 1
            if len(bucket["support_pairs_eq"]) < 3:
                bucket["support_pairs_eq"].append([left_id, right_id])
        else:
            bucket["neq_count"] += 1
            if len(bucket["support_pairs_neq"]) < 3:
                bucket["support_pairs_neq"].append([left_id, right_id])

    return evidence


def detect_macro_triangle_conflicts(
    golden_ids: list[str],
    evidence: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    eq_graph = nx.Graph()
    eq_graph.add_nodes_from(golden_ids)

    for pair, bucket in evidence.items():
        if int(bucket["eq_count"]) > 0:
            eq_graph.add_edge(*pair)

    conflicts: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for center in eq_graph.nodes():
        neighbors = sorted(str(v) for v in eq_graph.neighbors(center))
        if len(neighbors) < 2:
            continue
        for a, c in combinations(neighbors, 2):
            tri_key = tuple(sorted((a, str(center), c)))
            if tri_key in seen:
                continue
            seen.add(tri_key)
            ac_key = canonical_pair(a, c)
            ac_bucket = evidence.get(ac_key)
            if ac_bucket is None:
                continue
            if int(ac_bucket["neq_count"]) <= 0:
                continue
            conflicts.append(
                {
                    "golden_a": a,
                    "golden_b": str(center),
                    "golden_c": c,
                    "ac_eq_count": int(ac_bucket["eq_count"]),
                    "ac_neq_count": int(ac_bucket["neq_count"]),
                }
            )
    return conflicts


def stage4_global_conflict_resolving(
    predicted_df: pd.DataFrame,
    golden_records: dict[str, str],
    entity_to_golden: dict[str, str],
    out_dir: Path,
    model_name: str,
    enable_global_arbitration: bool,
    max_workers: int,
    max_retries: int,
    max_conflicts: int,
) -> dict[str, Any]:
    stage_dir = out_dir / "stage4_global_resolving"
    stage_dir.mkdir(parents=True, exist_ok=True)

    evidence = collect_macro_edge_evidence(predicted_df, entity_to_golden)
    golden_ids = sorted(golden_records.keys())

    edge_rows: list[dict[str, Any]] = []
    pair_conflicts: list[dict[str, Any]] = []
    for (left, right), bucket in sorted(evidence.items()):
        eq_count = int(bucket["eq_count"])
        neq_count = int(bucket["neq_count"])
        relation = "unknown"
        if eq_count > 0 and neq_count > 0:
            relation = "mixed"
        elif eq_count > 0:
            relation = "equal"
        elif neq_count > 0:
            relation = "not_equal"

        row = {
            "golden_left": left,
            "golden_right": right,
            "eq_count": eq_count,
            "neq_count": neq_count,
            "relation": relation,
            "support_pairs_eq": json.dumps(bucket["support_pairs_eq"], ensure_ascii=True),
            "support_pairs_neq": json.dumps(bucket["support_pairs_neq"], ensure_ascii=True),
        }
        edge_rows.append(row)
        if relation == "mixed":
            pair_conflicts.append(row)

    triangle_conflicts = detect_macro_triangle_conflicts(golden_ids, evidence)

    resolver = GlobalConflictResolver(
        model_name=model_name,
        max_retries=max_retries,
        cache_dir=dirty_matching_stage_cache_dir(model_name, "stage4"),
    )

    pair_judgments: list[dict[str, Any]] = []
    triangle_judgments: list[dict[str, Any]] = []
    arbitration_triggered = bool(pair_conflicts or triangle_conflicts)

    if arbitration_triggered and enable_global_arbitration:
        pair_jobs = pair_conflicts[: max(0, int(max_conflicts))]

        def run_pair(row: dict[str, Any]) -> dict[str, Any]:
            left = str(row["golden_left"])
            right = str(row["golden_right"])
            return resolver.resolve_pair(
                golden_a=left,
                golden_b=right,
                record_a=golden_records.get(left, ""),
                record_b=golden_records.get(right, ""),
            )

        pair_judgments = (
            thread_map(run_pair, pair_jobs, max_workers=max(1, int(max_workers)))
            if pair_jobs
            else []
        )

        tri_jobs = triangle_conflicts[: max(0, int(max_conflicts))]

        def run_triangle(row: dict[str, Any]) -> dict[str, Any]:
            golden_ids_triplet = [
                str(row["golden_a"]),
                str(row["golden_b"]),
                str(row["golden_c"]),
            ]
            return resolver.resolve_triangle(golden_ids_triplet, golden_records)

        triangle_judgments = (
            thread_map(run_triangle, tri_jobs, max_workers=max(1, int(max_workers)))
            if tri_jobs
            else []
        )

    uf = UnionFind(golden_ids)
    for row in edge_rows:
        if row["eq_count"] > 0 and row["neq_count"] == 0:
            uf.union(str(row["golden_left"]), str(row["golden_right"]))

    for judgment in pair_judgments:
        if not bool(judgment.get("match", False)):
            continue
        left, right = judgment["pair"]
        uf.union(str(left), str(right))

    for judgment in triangle_judgments:
        for cluster in judgment.get("clusters", []):
            members = [str(member) for member in cluster]
            if len(members) < 2:
                continue
            anchor = members[0]
            for member in members[1:]:
                uf.union(anchor, member)

    grouped_goldens: dict[str, list[str]] = {}
    for golden_id in golden_ids:
        root = uf.find(golden_id)
        grouped_goldens.setdefault(root, []).append(golden_id)

    golden_to_entities: dict[str, list[str]] = {}
    for entity_id, golden_id in entity_to_golden.items():
        golden_to_entities.setdefault(golden_id, []).append(entity_id)

    final_rows: list[dict[str, Any]] = []
    entity_rows: list[dict[str, Any]] = []
    for idx, golden_group in enumerate(
        sorted(grouped_goldens.values(), key=lambda x: (-len(x), x[0]))
    ):
        final_cluster_id = f"F{idx}"
        entities: list[str] = []
        for golden_id in sorted(golden_group):
            entities.extend(golden_to_entities.get(golden_id, []))
        entities = sorted(set(entities))

        final_rows.append(
            {
                "final_cluster_id": final_cluster_id,
                "golden_ids": json.dumps(sorted(golden_group), ensure_ascii=True),
                "entity_ids": json.dumps(entities, ensure_ascii=True),
                "size": int(len(entities)),
            }
        )
        for entity_id in entities:
            entity_rows.append({"entity_id": entity_id, "final_cluster_id": final_cluster_id})

    macro_edges_csv = stage_dir / "macro_edges.csv"
    pd.DataFrame(edge_rows).to_csv(macro_edges_csv, index=False)
    pair_conflicts_csv = stage_dir / "pair_conflicts.csv"
    pd.DataFrame(pair_conflicts).to_csv(pair_conflicts_csv, index=False)
    tri_conflicts_csv = stage_dir / "triangle_conflicts.csv"
    pd.DataFrame(triangle_conflicts).to_csv(tri_conflicts_csv, index=False)
    pair_judgments_csv = stage_dir / "pair_judgments.csv"
    pd.DataFrame(pair_judgments).to_csv(pair_judgments_csv, index=False)
    tri_judgments_csv = stage_dir / "triangle_judgments.csv"
    pd.DataFrame(triangle_judgments).to_csv(tri_judgments_csv, index=False)
    final_clusters_csv = stage_dir / "final_clusters.csv"
    pd.DataFrame(final_rows).to_csv(final_clusters_csv, index=False)
    entity_final_csv = stage_dir / "entity_final_cluster.csv"
    pd.DataFrame(entity_rows).to_csv(entity_final_csv, index=False)

    entity_to_final_cluster = {
        str(row["entity_id"]): str(row["final_cluster_id"]) for row in entity_rows
    }
    final_pair_df = predicted_df.copy()
    final_pair_df["final_cluster_id_left"] = final_pair_df["id_left"].map(
        lambda x: entity_to_final_cluster.get(str(x), "")
    )
    final_pair_df["final_cluster_id_right"] = final_pair_df["id_right"].map(
        lambda x: entity_to_final_cluster.get(str(x), "")
    )
    final_pair_df["pred_final"] = (
        final_pair_df["final_cluster_id_left"]
        == final_pair_df["final_cluster_id_right"]
    )

    final_pair_predictions_csv = stage_dir / "final_pair_predictions.csv"
    final_pair_df.to_csv(final_pair_predictions_csv, index=False)

    baseline_pred = final_pair_df["pred"].astype(bool) if "pred" in final_pair_df.columns else None

    pair_metrics: dict[str, Any] = {
        "num_pairs": int(len(final_pair_df)),
        "positive_predictions_final": int(final_pair_df["pred_final"].sum()),
        "positive_predictions_baseline": int(baseline_pred.sum())
        if baseline_pred is not None
        else None,
        "changed_pairs": int((final_pair_df["pred_final"] != baseline_pred).sum())
        if baseline_pred is not None
        else None,
        "flipped_to_true": int((final_pair_df["pred_final"] & (~baseline_pred)).sum())
        if baseline_pred is not None
        else None,
        "flipped_to_false": int(((~final_pair_df["pred_final"]) & baseline_pred).sum())
        if baseline_pred is not None
        else None,
    }

    if "label" in final_pair_df.columns:
        labels = final_pair_df["label"].astype(bool).tolist()
        preds_final = final_pair_df["pred_final"].astype(bool).tolist()
        final_report = classification_report(
            labels,
            preds_final,
            digits=4,
            output_dict=True,
            zero_division=0,
        )
        final_matrix = confusion_matrix(labels, preds_final, labels=[False, True]).tolist()

        pair_metrics["positive_labels"] = int(sum(labels))
        pair_metrics["classification_report_final"] = final_report
        pair_metrics["confusion_matrix_final"] = {
            "labels": [False, True],
            "matrix": final_matrix,
        }

        if baseline_pred is not None:
            preds_base = baseline_pred.tolist()
            base_report = classification_report(
                labels,
                preds_base,
                digits=4,
                output_dict=True,
                zero_division=0,
            )
            base_matrix = confusion_matrix(labels, preds_base, labels=[False, True]).tolist()
            pair_metrics["classification_report_baseline"] = base_report
            pair_metrics["confusion_matrix_baseline"] = {
                "labels": [False, True],
                "matrix": base_matrix,
            }

    pair_metrics_json = stage_dir / "pair_metrics.json"
    with pair_metrics_json.open("w", encoding="utf-8") as f:
        json.dump(pair_metrics, f, ensure_ascii=True, indent=2)

    stats = {
        "golden_nodes": int(len(golden_ids)),
        "macro_edges": int(len(edge_rows)),
        "pair_conflicts": int(len(pair_conflicts)),
        "triangle_conflicts": int(len(triangle_conflicts)),
        "arbitration_triggered": bool(arbitration_triggered),
        "arbitration_enabled": bool(enable_global_arbitration),
        "pair_judgments": int(len(pair_judgments)),
        "triangle_judgments": int(len(triangle_judgments)),
        "final_clusters": int(len(final_rows)),
        "api_cost": float(
            resolver.cost if arbitration_triggered and enable_global_arbitration else 0.0
        ),
        "pair_level_metrics": pair_metrics,
    }
    summary_json = stage_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=True, indent=2)

    return {
        "stats": stats,
        "files": {
            "macro_edges_csv": str(macro_edges_csv),
            "pair_conflicts_csv": str(pair_conflicts_csv),
            "triangle_conflicts_csv": str(tri_conflicts_csv),
            "pair_judgments_csv": str(pair_judgments_csv),
            "triangle_judgments_csv": str(tri_judgments_csv),
            "final_clusters_csv": str(final_clusters_csv),
            "entity_final_csv": str(entity_final_csv),
            "final_pair_predictions_csv": str(final_pair_predictions_csv),
            "pair_metrics_json": str(pair_metrics_json),
            "summary_json": str(summary_json),
        },
    }
