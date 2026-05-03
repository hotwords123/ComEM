from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from src.louvain_community import (
    detect_recursive_louvain_communities,
    visualize_html as visualize_louvain_html,
    visualize_png as visualize_louvain_png,
)


def stage1_filter_and_split(
    graph: nx.Graph,
    out_dir: Path,
    resolution: float,
    seed: int,
    max_community_size: int,
    max_recursion_depth: int,
    resolution_scale: float,
) -> dict[str, Any]:
    stage_dir = out_dir / "stage1_partition"
    stage_dir.mkdir(parents=True, exist_ok=True)

    if graph.number_of_nodes() == 0:
        empty_stats = {
            "nodes": 0,
            "edges": 0,
            "communities": 0,
            "modularity": 0.0,
            "recursive": {
                "enabled": True,
                "max_community_size": int(max_community_size),
                "max_depth": int(max_recursion_depth),
                "base_resolution": float(resolution),
                "resolution_scale": float(resolution_scale),
            },
        }
        summary_path = stage_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(empty_stats, f, ensure_ascii=True, indent=2)
        return {
            "communities": [],
            "node_to_community": {},
            "stats": empty_stats,
            "files": {"summary_json": str(summary_path)},
        }

    communities, node_to_community, modularity, recursive_stats = (
        detect_recursive_louvain_communities(
            graph,
            resolution=resolution,
            seed=seed,
            max_community_size=max_community_size,
            max_depth=max_recursion_depth,
            resolution_scale=resolution_scale,
        )
    )

    rows = []
    for node in sorted(str(node) for node in graph.nodes()):
        rows.append(
            {
                "entity_id": node,
                "community_id": int(node_to_community.get(node, -1)),
                "degree": int(graph.degree[node]),
            }
        )
    communities_csv = stage_dir / "communities.csv"
    pd.DataFrame(rows).to_csv(communities_csv, index=False)

    sizes = sorted((len(c) for c in communities), reverse=True)
    stats = {
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
        "communities": int(len(communities)),
        "community_sizes_top10": [int(v) for v in sizes[:10]],
        "modularity": float(modularity),
        "recursive": recursive_stats,
    }
    summary_path = stage_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=True, indent=2)

    png_path = stage_dir / "louvain_partition.png"
    html_path = stage_dir / "louvain_partition.html"
    title = "Stage1 Recursive Louvain Partition"
    visualize_louvain_png(
        graph=graph,
        node_to_community=node_to_community,
        out_png=png_path,
        title=title,
        seed=seed,
    )
    visualize_louvain_html(
        graph=graph,
        node_to_community=node_to_community,
        out_html=html_path,
        title=title,
        seed=seed,
    )

    return {
        "communities": communities,
        "node_to_community": node_to_community,
        "stats": stats,
        "files": {
            "communities_csv": str(communities_csv),
            "summary_json": str(summary_path),
            "graph_png": str(png_path),
            "graph_html": str(html_path),
        },
    }
