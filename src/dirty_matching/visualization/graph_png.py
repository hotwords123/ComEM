from __future__ import annotations

from pathlib import Path

import networkx as nx
from matplotlib import pyplot as plt


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
