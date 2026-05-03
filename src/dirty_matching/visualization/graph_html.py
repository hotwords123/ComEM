from __future__ import annotations

import html
import math
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from src.dirty_matching.core.lookups import (
    build_edge_outcome_lookup,
    build_entity_cluster_lookup,
    build_entity_record_lookup,
    cluster_internal_layout,
    quantile,
    truncate_text,
)


def layout_nodes_by_gt(
    nodes: list[str],
    predicted_df: pd.DataFrame,
    graph: nx.Graph,
    edge_outcomes: dict[tuple[str, str], str],
    cluster_layout_k: float | None,
    cluster_layout_iterations: int,
    cluster_layout_spread: float | None,
    cluster_layout_norm_quantile: float,
) -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    cluster_lookup = build_entity_cluster_lookup(predicted_df)
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
        if (
            left_cluster is None
            or right_cluster is None
            or left_cluster == right_cluster
        ):
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
        median_x = quantile(xs, 0.5)
        median_y = quantile(ys, 0.5)
        shifted = [(x - median_x, y - median_y) for x, y in center_pos.values()]
        radii = [max(abs(x), abs(y)) for x, y in shifted]
        normalizer = quantile(radii, cluster_layout_norm_quantile)
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
        local_pos = cluster_internal_layout(cluster_nodes, subgraph)

        scale = max(0.3, min(1.6, 0.28 * math.sqrt(len(cluster_nodes))))
        for node in cluster_nodes:
            x, y = local_pos[node]
            pos[node] = (center_x + scale * x, center_y + scale * y)

    return pos, node_cluster_map


def cluster_color_map(cluster_ids: list[str]) -> dict[str, str]:
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


def build_edge_trace(
    pos: dict[str, tuple[float, float]],
    edges: list[tuple[str, str]],
    name: str,
    color: str,
    width: float,
    dash: str = "solid",
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
        hoverinfo="none",
        name=name,
        showlegend=True,
    )


def visualize_graph_html(
    graph: nx.Graph,
    predicted_df: pd.DataFrame,
    out_html: Path,
    title: str,
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
    edge_outcomes = build_edge_outcome_lookup(predicted_df)
    pos, node_cluster_map = layout_nodes_by_gt(
        nodes,
        predicted_df,
        graph,
        edge_outcomes,
        cluster_layout_k=cluster_layout_k,
        cluster_layout_iterations=cluster_layout_iterations,
        cluster_layout_spread=cluster_layout_spread,
        cluster_layout_norm_quantile=cluster_layout_norm_quantile,
    )
    cluster_color_lookup = cluster_color_map(list(set(node_cluster_map.values())))

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

    fig.add_trace(build_edge_trace(pos, tp_edges, f"TP ({len(tp_edges)})", "#27ae60", 1.15))
    fig.add_trace(build_edge_trace(pos, fn_edges, f"FN ({len(fn_edges)})", "#f39c12", 1.15))
    fig.add_trace(
        build_edge_trace(pos, fp_edges, f"FP ({len(fp_edges)})", "#e74c3c", 1.5, dash="dash")
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

    record_lookup = build_entity_record_lookup(predicted_df)

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
        rec = truncate_text(record_lookup.get(str(node), ""))
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
        title={"text": title, "x": 0.5, "xanchor": "center", "y": 0.98, "yanchor": "top"},
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 60, "b": 80},
        xaxis={"visible": False},
        yaxis={"visible": False},
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.08,
            "xanchor": "center",
            "x": 0.5,
        },
    )
    fig.write_html(out_html, include_plotlyjs="cdn")
