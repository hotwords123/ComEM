from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd


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
