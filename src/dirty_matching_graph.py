from __future__ import annotations

import argparse
import html
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.contrib.concurrent import thread_map

from src.dirty_blocking import run_dirty_pipeline
from src.matching import Matching


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

    def _run_instance(instance: dict[str, Any]) -> tuple[list[int], list[bool]]:
        preds = matcher(instance)
        return instance["row_indexes"], preds

    mapped = thread_map(_run_instance, instances, max_workers=max_workers)
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

    report = classification_report(labels, preds, digits=4, output_dict=True, zero_division=0)
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


def build_matching_graph(predicted_df: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()

    nodes = set(predicted_df["id_left"].tolist()) | set(predicted_df["id_right"].tolist())
    graph.add_nodes_from(nodes)

    matched = predicted_df[predicted_df["pred"]]
    graph.add_edges_from(matched[["id_left", "id_right"]].itertuples(index=False, name=None))
    return graph


def graph_statistics(graph: nx.Graph) -> dict[str, Any]:
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    components = list(nx.connected_components(graph))
    component_sizes = sorted((len(c) for c in components), reverse=True)
    avg_degree = 0.0 if nodes == 0 else (2.0 * edges) / nodes

    return {
        "nodes": int(nodes),
        "edges": int(edges),
        "connected_components": int(len(components)),
        "largest_component_size": int(component_sizes[0]) if component_sizes else 0,
        "average_degree": float(avg_degree),
        "component_sizes_top10": [int(v) for v in component_sizes[:10]],
    }


def visualize_graph_png(graph: nx.Graph, out_png: Path, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 10))
    if graph.number_of_nodes() == 0:
        plt.title(f"{title} (empty graph)")
        plt.axis("off")
        plt.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close()
        return

    pos = nx.spring_layout(graph, seed=42)
    nx.draw_networkx_edges(graph, pos, alpha=0.35, width=0.7, edge_color="#7f8c8d")
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=35,
        node_color="#2e86de",
        alpha=0.85,
        linewidths=0,
    )
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def _truncate_text(text: str, max_chars: int = 240) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 3]}..."


def _build_entity_record_lookup(predicted_df: pd.DataFrame) -> dict[str, str]:
    record_lookup: dict[str, str] = {}
    for left, right, record_left, record_right in predicted_df[
        ["id_left", "id_right", "record_left", "record_right"]
    ].itertuples(index=False, name=None):
        record_lookup[str(left)] = str(record_left)
        record_lookup[str(right)] = str(record_right)
    return record_lookup


def _build_edge_outcome_lookup(predicted_df: pd.DataFrame) -> dict[tuple[str, str], str]:
    outcome_lookup: dict[tuple[str, str], str] = {}
    for left, right, pred, label in predicted_df[
        ["id_left", "id_right", "pred", "label"]
    ].itertuples(index=False, name=None):
        key = tuple(sorted((str(left), str(right))))
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


def _cluster_internal_layout(nodes: list[str], subgraph: nx.Graph) -> dict[str, tuple[float, float]]:
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


def _quantile(values: list[float], q: float) -> float:
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


def _layout_nodes_by_gt(
    nodes: list[str],
    predicted_df: pd.DataFrame,
    graph: nx.Graph,
    edge_outcomes: dict[tuple[str, str], str],
    cluster_layout_k: float | None,
    cluster_layout_iterations: int,
    cluster_layout_spread: float | None,
    cluster_layout_norm_quantile: float,
) -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    cluster_lookup = _build_entity_cluster_lookup(predicted_df)
    groups: dict[str, list[str]] = {}
    node_cluster_map: dict[str, str] = {}
    for node in nodes:
        cluster_id = cluster_lookup.get(str(node), "unknown")
        groups.setdefault(cluster_id, []).append(str(node))
        node_cluster_map[str(node)] = cluster_id

    sorted_clusters = sorted(groups.keys(), key=lambda x: (x == "unknown", x))
    center_graph = nx.Graph()
    center_graph.add_nodes_from(sorted_clusters)

    for (left, right), outcome in edge_outcomes.items():
        left_cluster = node_cluster_map.get(left)
        right_cluster = node_cluster_map.get(right)
        if left_cluster is None or right_cluster is None or left_cluster == right_cluster:
            continue

        # Emphasize cross-cluster false positives to expose misclassification corridors.
        weight = 0.0
        if outcome == "FP":
            weight = 3.0
        elif outcome == "FN":
            weight = 1.0
        elif outcome == "TP":
            weight = 0.2
        if weight <= 0.0:
            continue

        if center_graph.has_edge(left_cluster, right_cluster):
            center_graph[left_cluster][right_cluster]["weight"] += weight
        else:
            center_graph.add_edge(left_cluster, right_cluster, weight=weight)

    n_clusters = max(1, len(sorted_clusters))
    effective_k = (
        float(cluster_layout_k)
        if cluster_layout_k is not None
        else max(0.8, 2.5 / math.sqrt(n_clusters))
    )
    if center_graph.number_of_edges() > 0:
        center_pos = nx.spring_layout(
            center_graph,
            seed=42,
            weight="weight",
            k=effective_k,
            iterations=max(50, int(cluster_layout_iterations)),
        )
    else:
        center_pos = {}

    if not center_pos:
        cols = max(1, int(math.ceil(math.sqrt(n_clusters))))
        spacing = 4.0
        center_pos = {
            cluster_id: (
                (idx % cols - (cols - 1) / 2.0) * spacing,
                (-(idx // cols) + (math.ceil(n_clusters / cols) - 1) / 2.0) * spacing,
            )
            for idx, cluster_id in enumerate(sorted_clusters)
        }
    else:
        xs = [float(x) for x, _ in center_pos.values()]
        ys = [float(y) for _, y in center_pos.values()]
        median_x = _quantile(xs, 0.5)
        median_y = _quantile(ys, 0.5)
        shifted = [(x - median_x, y - median_y) for x, y in center_pos.values()]
        radii = [max(abs(x), abs(y)) for x, y in shifted]
        normalizer = _quantile(radii, cluster_layout_norm_quantile)
        normalizer = max(1e-6, float(normalizer))
        scale = (
            float(cluster_layout_spread)
            if cluster_layout_spread is not None
            else max(4.0, 0.95 * n_clusters)
        )
        center_pos = {
            cluster_id: (
                scale * max(-1.4, min(1.4, float(x - median_x) / normalizer)),
                scale * max(-1.4, min(1.4, float(y - median_y) / normalizer)),
            )
            for cluster_id, (x, y) in center_pos.items()
        }

    pos: dict[str, tuple[float, float]] = {}
    for cluster_id in sorted_clusters:
        center_x, center_y = center_pos.get(cluster_id, (0.0, 0.0))
        cluster_nodes = sorted(groups[cluster_id])
        subgraph = graph.subgraph(cluster_nodes).copy()
        local_pos = _cluster_internal_layout(cluster_nodes, subgraph)

        scale = max(0.3, min(1.6, 0.28 * math.sqrt(len(cluster_nodes))))
        for node in cluster_nodes:
            x, y = local_pos[node]
            pos[node] = (center_x + scale * x, center_y + scale * y)

    return pos, node_cluster_map


def _cluster_color_map(cluster_ids: list[str]) -> dict[str, str]:
    palette = [
        "#1f77b4",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#17becf",
        "#e377c2",
        "#8c564b",
        "#bcbd22",
        "#7f7f7f",
        "#d62728",
    ]
    color_map: dict[str, str] = {}
    sorted_ids = sorted(cluster_ids, key=lambda x: (x == "unknown", x))
    for idx, cluster_id in enumerate(sorted_ids):
        if cluster_id == "unknown":
            color_map[cluster_id] = "#7f8c8d"
        else:
            color_map[cluster_id] = palette[idx % len(palette)]
    return color_map


def _build_edge_trace(
    pos: dict[str, tuple[float, float]],
    edges: list[tuple[str, str]],
    name: str,
    color: str,
    width: float,
    dash: str = "solid",
    opacity: float = 1.0,
) -> go.Scatter:
    edge_x: list[float] = []
    edge_y: list[float] = []
    for left, right in edges:
        x0, y0 = pos[left]
        x1, y1 = pos[right]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    return go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line={"width": width, "color": color, "dash": dash},
        opacity=max(0.0, min(1.0, float(opacity))),
        hoverinfo="none",
        name=name,
        showlegend=True,
    )


def _edge_focus_menu(dimmed_edge_opacity: float) -> list[dict[str, Any]]:
    dim = max(0.0, min(1.0, float(dimmed_edge_opacity)))
    return [
        {
            "type": "buttons",
            "direction": "right",
            "x": 0.0,
            "y": 1.14,
            "xanchor": "left",
            "yanchor": "top",
            "showactive": True,
            "buttons": [
                {
                    "label": "All Edges",
                    "method": "restyle",
                    "args": [{"opacity": [1.0, 1.0, 1.0]}, [0, 1, 2]],
                },
                {
                    "label": "Focus TP",
                    "method": "restyle",
                    "args": [{"opacity": [1.0, dim, dim]}, [0, 1, 2]],
                },
                {
                    "label": "Focus FN",
                    "method": "restyle",
                    "args": [{"opacity": [dim, 1.0, dim]}, [0, 1, 2]],
                },
                {
                    "label": "Focus FP",
                    "method": "restyle",
                    "args": [{"opacity": [dim, dim, 1.0]}, [0, 1, 2]],
                },
            ],
        }
    ]


def visualize_graph_html(
    graph: nx.Graph,
    predicted_df: pd.DataFrame,
    out_html: Path,
    title: str,
    dimmed_edge_opacity: float = 0.05,
    cluster_layout_k: float | None = None,
    cluster_layout_iterations: int = 350,
    cluster_layout_spread: float | None = None,
    cluster_layout_norm_quantile: float = 0.9,
) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)

    fig = go.Figure()
    if graph.number_of_nodes() == 0:
        fig.update_layout(title=title, template="plotly_white")
        fig.write_html(out_html, include_plotlyjs="cdn")
        return

    nodes = sorted(str(node) for node in graph.nodes())
    edge_outcomes = _build_edge_outcome_lookup(predicted_df)
    pos, node_cluster_map = _layout_nodes_by_gt(
        nodes,
        predicted_df,
        graph,
        edge_outcomes,
        cluster_layout_k=cluster_layout_k,
        cluster_layout_iterations=cluster_layout_iterations,
        cluster_layout_spread=cluster_layout_spread,
        cluster_layout_norm_quantile=cluster_layout_norm_quantile,
    )
    cluster_color_lookup = _cluster_color_map(list(set(node_cluster_map.values())))

    tp_edges: list[tuple[str, str]] = []
    fp_edges: list[tuple[str, str]] = []
    fn_edges: list[tuple[str, str]] = []
    for pair, outcome in edge_outcomes.items():
        left, right = pair
        if left not in pos or right not in pos:
            continue
        if outcome == "TP":
            tp_edges.append((left, right))
        elif outcome == "FP":
            fp_edges.append((left, right))
        elif outcome == "FN":
            fn_edges.append((left, right))

    fig.add_trace(
        _build_edge_trace(
            pos,
            tp_edges,
            f"TP GT+ solid ({len(tp_edges)})",
            "#27ae60",
            1.15,
            opacity=1.0,
        )
    )
    fig.add_trace(
        _build_edge_trace(
            pos,
            fn_edges,
            f"FN GT+ solid ({len(fn_edges)})",
            "#f39c12",
            1.15,
            opacity=1.0,
        )
    )
    fig.add_trace(
        _build_edge_trace(
            pos,
            fp_edges,
            f"FP dashed ({len(fp_edges)})",
            "#e74c3c",
            1.5,
            dash="dash",
            opacity=1.0,
        )
    )

    node_tp_degree = {node: 0 for node in nodes}
    node_fp_degree = {node: 0 for node in nodes}
    node_fn_degree = {node: 0 for node in nodes}
    for left, right in tp_edges:
        node_tp_degree[left] += 1
        node_tp_degree[right] += 1
    for left, right in fp_edges:
        node_fp_degree[left] += 1
        node_fp_degree[right] += 1
    for left, right in fn_edges:
        node_fn_degree[left] += 1
        node_fn_degree[right] += 1

    record_lookup = _build_entity_record_lookup(predicted_df)

    node_x: list[float] = []
    node_y: list[float] = []
    node_text: list[str] = []
    node_colors: list[str] = []
    for node in nodes:
        x, y = pos[str(node)]
        node_x.append(x)
        node_y.append(y)
        cluster_id = node_cluster_map.get(str(node), "unknown")
        node_colors.append(cluster_color_lookup.get(cluster_id, "#7f8c8d"))
        degree = int(graph.degree[str(node)])
        rec = _truncate_text(record_lookup.get(str(node), ""))
        node_text.append(
            "<br>".join(
                [
                    f"id={html.escape(str(node))}",
                    f"gt_cluster={html.escape(str(cluster_id))}",
                    f"degree={degree}",
                    f"tp_edges={node_tp_degree[str(node)]}",
                    f"fp_edges={node_fp_degree[str(node)]}",
                    f"fn_edges={node_fn_degree[str(node)]}",
                    f"record={html.escape(rec)}",
                ]
            )
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker={
            "size": 9,
            "color": node_colors,
            "line": {"width": 0.5, "color": "#2c3e50"},
        },
        name="Nodes (color by GT cluster)",
        showlegend=True,
    )

    fig.add_trace(node_trace)
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        xaxis={"visible": False},
        yaxis={"visible": False},
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        updatemenus=_edge_focus_menu(dimmed_edge_opacity),
    )
    fig.write_html(out_html, include_plotlyjs="cdn")


def _build_pair_prediction_lookup(predicted_df: pd.DataFrame) -> dict[tuple[str, str], bool]:
    lookup: dict[tuple[str, str], bool] = {}
    for left, right, pred in predicted_df[["id_left", "id_right", "pred"]].itertuples(
        index=False,
        name=None,
    ):
        key = tuple(sorted((str(left), str(right))))
        lookup[key] = bool(pred)
    return lookup


def _build_entity_cluster_lookup(predicted_df: pd.DataFrame) -> dict[str, str]:
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


def _gt_partition_for_triplet(
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


def _collect_mode_violations(
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
                gt_info = _gt_partition_for_triplet(a, str(center), c, cluster_lookup)
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
    pair_lookup = _build_pair_prediction_lookup(predicted_df)
    cluster_lookup = _build_entity_cluster_lookup(predicted_df)

    requested_violations, requested_stats = _collect_mode_violations(
        graph,
        pair_lookup,
        cluster_lookup,
        mode,
    )
    strict_violations, strict_stats = _collect_mode_violations(
        graph,
        pair_lookup,
        cluster_lookup,
        "strict_negative",
    )
    include_violations, include_stats = _collect_mode_violations(
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


def load_or_generate_candidates(
    dataset_name: str,
    reader_root: Path,
    candidates_csv: Path | None,
    topk: int,
    force_rebuild_index: bool,
    sample_frac: float | None,
    sample_n: int | None,
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
        sample_seed=sample_seed,
    )


def run_default_cora_pipeline() -> dict[str, Any]:
    return run_pipeline(
        dataset_name="cora",
        reader_root=Path("data/pyJedAI/data/der/cora"),
        model_name="gpt-4o-mini",
        output_dir=Path("results/dirty_matching/cora"),
        candidates_csv=None,
        topk=20,
        force_rebuild_index=True,
        sample_frac=None,
        sample_n=200,
        sample_seed=42,
        max_workers=16,
        violation_mode="strict_negative",
        dimmed_edge_opacity=0.05,
        cluster_layout_k=None,
        cluster_layout_iterations=350,
        cluster_layout_spread=None,
        cluster_layout_norm_quantile=0.9,
    )


def run_pipeline(
    dataset_name: str,
    reader_root: Path,
    model_name: str,
    output_dir: Path,
    candidates_csv: Path | None,
    topk: int,
    force_rebuild_index: bool,
    sample_frac: float | None,
    sample_n: int | None,
    sample_seed: int,
    max_workers: int,
    violation_mode: str,
    dimmed_edge_opacity: float,
    cluster_layout_k: float | None,
    cluster_layout_iterations: int,
    cluster_layout_spread: float | None,
    cluster_layout_norm_quantile: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_or_generate_candidates(
        dataset_name=dataset_name,
        reader_root=reader_root,
        candidates_csv=candidates_csv,
        topk=topk,
        force_rebuild_index=force_rebuild_index,
        sample_frac=sample_frac,
        sample_n=sample_n,
        sample_seed=sample_seed,
    )

    candidates_out = output_dir / "candidates.csv"
    candidates.to_csv(candidates_out, index=False)

    predicted_df, metrics = predict_candidates(
        candidates=candidates,
        model_name=model_name,
        max_workers=max_workers,
    )
    predicted_out = output_dir / "predictions.csv"
    predicted_df.to_csv(predicted_out, index=False)

    graph = build_matching_graph(predicted_df)
    graph_stats = graph_statistics(graph)

    png_out = output_dir / "matching_graph.png"
    html_out = output_dir / "matching_graph.html"
    visualize_graph_png(graph, png_out, title=f"{dataset_name} matching graph ({model_name})")
    visualize_graph_html(
        graph,
        predicted_df,
        html_out,
        title=f"{dataset_name} matching graph ({model_name})",
        dimmed_edge_opacity=dimmed_edge_opacity,
        cluster_layout_k=cluster_layout_k,
        cluster_layout_iterations=cluster_layout_iterations,
        cluster_layout_spread=cluster_layout_spread,
        cluster_layout_norm_quantile=cluster_layout_norm_quantile,
    )

    violations_df, violation_stats, comparison_stats = detect_transitivity_violations(
        graph=graph,
        predicted_df=predicted_df,
        mode=violation_mode,
    )
    violations_out = output_dir / f"transitivity_violations_{violation_mode}.csv"
    violations_df.to_csv(violations_out, index=False)

    summary = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "topk": topk,
        "sample_frac": sample_frac,
        "sample_n": sample_n,
        "sample_seed": sample_seed,
        "visualization": {
            "dimmed_edge_opacity": dimmed_edge_opacity,
            "cluster_layout_k": cluster_layout_k,
            "cluster_layout_iterations": cluster_layout_iterations,
            "cluster_layout_spread": cluster_layout_spread,
            "cluster_layout_norm_quantile": cluster_layout_norm_quantile,
        },
        "metrics": metrics,
        "graph": graph_stats,
        "transitivity": {
            "selected_mode": violation_stats,
            "comparison": comparison_stats,
            "violations_csv": str(violations_out),
        },
        "files": {
            "candidates_csv": str(candidates_out),
            "predictions_csv": str(predicted_out),
            "graph_png": str(png_out),
            "graph_html": str(html_out),
            "violations_csv": str(violations_out),
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    print("Run completed.")
    print(json.dumps(summary["graph"], ensure_ascii=True, indent=2))
    print(json.dumps(summary["transitivity"], ensure_ascii=True, indent=2))
    print(f"API cost: {metrics['api_cost']:.4f}")
    print(f"Outputs written to: {output_dir}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run dirty ER matching evaluation, build a match graph, and audit transitivity violations."
        )
    )
    parser.add_argument("--dataset-name", type=str, default="cora")
    parser.add_argument(
        "--reader-root",
        type=Path,
        default=Path("data/pyJedAI/data/der/cora"),
    )
    parser.add_argument("--candidates-csv", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default="gpt-4o-mini")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument(
        "--violation-mode",
        type=str,
        choices=["strict_negative", "include_missing"],
        default="strict_negative",
    )

    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--sample-n", type=int, default=200)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--dimmed-edge-opacity",
        type=float,
        default=0.05,
        help="Opacity for de-emphasized layers when using HTML focus buttons.",
    )
    parser.add_argument(
        "--cluster-layout-k",
        type=float,
        default=None,
        help="Spring-layout ideal distance for GT-cluster centers.",
    )
    parser.add_argument(
        "--cluster-layout-iterations",
        type=int,
        default=350,
        help="Iterations for GT-cluster center spring-layout.",
    )
    parser.add_argument(
        "--cluster-layout-spread",
        type=float,
        default=None,
        help="Global spread scale for cluster centers after normalization.",
    )
    parser.add_argument(
        "--cluster-layout-norm-quantile",
        type=float,
        default=0.9,
        help="Quantile used to robustly normalize cluster-center distances.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dirty_matching/cora"),
    )
    parser.add_argument(
        "--force-rebuild-index",
        action="store_true",
        help="Force rebuilding SparseRetriever index before searching.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        dataset_name=args.dataset_name,
        reader_root=args.reader_root,
        model_name=args.model_name,
        output_dir=args.output_dir,
        candidates_csv=args.candidates_csv,
        topk=args.topk,
        force_rebuild_index=args.force_rebuild_index,
        sample_frac=args.sample_frac,
        sample_n=args.sample_n,
        sample_seed=args.sample_seed,
        max_workers=args.max_workers,
        violation_mode=args.violation_mode,
        dimmed_edge_opacity=args.dimmed_edge_opacity,
        cluster_layout_k=args.cluster_layout_k,
        cluster_layout_iterations=args.cluster_layout_iterations,
        cluster_layout_spread=args.cluster_layout_spread,
        cluster_layout_norm_quantile=args.cluster_layout_norm_quantile,
    )
