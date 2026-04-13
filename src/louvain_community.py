from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex


def _parse_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))

    text = str(value).strip().lower()
    if text in {"true", "1", "t", "yes", "y"}:
        return True
    if text in {"false", "0", "f", "no", "n", ""}:
        return False

    raise ValueError(f"Cannot parse boolean value: {value}")


def load_predictions(predictions_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(predictions_csv)

    required_cols = {"id_left", "id_right", "pred"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in predictions CSV: {sorted(missing)}"
        )

    out = df.copy()
    out["id_left"] = out["id_left"].astype(str)
    out["id_right"] = out["id_right"].astype(str)
    out["pred"] = out["pred"].map(_parse_bool)
    if "label" in out.columns:
        out["label"] = out["label"].map(_parse_bool)
    return out


def build_graph(df: pd.DataFrame, edge_column: str) -> nx.Graph:
    if edge_column not in df.columns:
        raise ValueError(f"edge_column '{edge_column}' not found in CSV")

    graph = nx.Graph()
    nodes = set(df["id_left"].tolist()) | set(df["id_right"].tolist())
    graph.add_nodes_from(nodes)

    edges_df = df[df[edge_column].map(_parse_bool)][["id_left", "id_right"]].copy()
    edges_df["src"] = edges_df[["id_left", "id_right"]].min(axis=1)
    edges_df["dst"] = edges_df[["id_left", "id_right"]].max(axis=1)
    edges = (
        edges_df[["src", "dst"]].drop_duplicates().itertuples(index=False, name=None)
    )
    graph.add_edges_from(edges)
    return graph


def detect_louvain_communities(
    graph: nx.Graph,
    resolution: float,
    seed: int,
) -> tuple[list[set[str]], dict[str, int], float]:
    if graph.number_of_nodes() == 0:
        return [], {}, 0.0

    communities_raw = nx.community.louvain_communities(
        graph,
        resolution=resolution,
        seed=seed,
    )
    communities = [
        set(str(node) for node in community) for community in communities_raw
    ]
    communities.sort(key=len, reverse=True)

    node_to_community: dict[str, int] = {}
    for cid, community in enumerate(communities):
        for node in community:
            node_to_community[node] = cid

    if graph.number_of_edges() == 0:
        modularity = 0.0
    else:
        modularity = float(nx.community.modularity(graph, communities))

    return communities, node_to_community, modularity


def detect_recursive_louvain_communities(
    graph: nx.Graph,
    resolution: float,
    seed: int,
    max_community_size: int,
    max_depth: int,
    resolution_scale: float,
) -> tuple[list[set[str]], dict[str, int], float, dict[str, Any]]:
    if max_community_size < 1:
        raise ValueError("max_community_size must be >= 1")
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")
    if resolution_scale <= 1.0:
        raise ValueError("resolution_scale must be > 1.0")
    if graph.number_of_nodes() == 0:
        raise ValueError("Input graph must have at least one node for community detection")

    split_attempts = 0
    split_successes = 0
    stopped_by_depth_limit = 0
    stopped_by_no_split = 0
    call_counter = 0

    def _next_seed(depth: int) -> int:
        nonlocal call_counter
        call_counter += 1
        return int(seed + depth * 1009 + call_counter)

    def _recursive_split(
        nodes: set[str],
        depth: int,
        local_resolution: float,
    ) -> list[set[str]]:
        nonlocal split_attempts
        nonlocal split_successes
        nonlocal stopped_by_depth_limit
        nonlocal stopped_by_no_split

        if len(nodes) <= max_community_size:
            return [nodes]
        if depth >= max_depth:
            stopped_by_depth_limit += 1
            return [nodes]

        split_attempts += 1
        subgraph = graph.subgraph(nodes).copy()
        next_resolution = float(local_resolution * resolution_scale)
        sub_communities_raw = nx.community.louvain_communities(
            subgraph,
            resolution=next_resolution,
            seed=_next_seed(depth),
        )
        sub_communities = [
            set(str(node) for node in community) for community in sub_communities_raw
        ]

        if len(sub_communities) <= 1:
            stopped_by_no_split += 1
            return [nodes]

        split_successes += 1
        leaves: list[set[str]] = []
        for community in sub_communities:
            leaves.extend(_recursive_split(community, depth + 1, next_resolution))
        return leaves

    root_communities_raw = nx.community.louvain_communities(
        graph,
        resolution=resolution,
        seed=_next_seed(0),
    )
    root_communities = [
        set(str(node) for node in community) for community in root_communities_raw
    ]

    final_communities: list[set[str]] = []
    for community in root_communities:
        final_communities.extend(_recursive_split(community, 0, resolution))
    final_communities.sort(key=len, reverse=True)

    node_to_community: dict[str, int] = {}
    for cid, community in enumerate(final_communities):
        for node in community:
            node_to_community[node] = cid

    if graph.number_of_edges() == 0 or not final_communities:
        modularity = 0.0
    else:
        modularity = float(nx.community.modularity(graph, final_communities))

    stats = {
        "enabled": True,
        "max_community_size": int(max_community_size),
        "max_depth": int(max_depth),
        "base_resolution": float(resolution),
        "resolution_scale": float(resolution_scale),
        "root_communities": int(len(root_communities)),
        "split_attempts": int(split_attempts),
        "split_successes": int(split_successes),
        "stopped_by_depth_limit": int(stopped_by_depth_limit),
        "stopped_by_no_split": int(stopped_by_no_split),
        "max_leaf_size": int(max((len(c) for c in final_communities), default=0)),
    }
    return final_communities, node_to_community, modularity, stats


def _community_color_map(community_ids: list[int]) -> dict[int, str]:
    if not community_ids:
        return {}

    unique_ids = sorted(set(community_ids))
    if len(unique_ids) <= 20:
        cmap = plt.get_cmap("tab20")
        return {cid: to_hex(cmap(idx % 20)) for idx, cid in enumerate(unique_ids)}

    cmap = plt.get_cmap("hsv")
    denom = max(1, len(unique_ids))
    return {cid: to_hex(cmap(idx / denom)) for idx, cid in enumerate(unique_ids)}


def visualize_png(
    graph: nx.Graph,
    node_to_community: dict[str, int],
    out_png: Path,
    title: str,
    seed: int,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 11))
    if graph.number_of_nodes() == 0:
        plt.title(f"{title} (empty graph)")
        plt.axis("off")
        plt.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close()
        return

    pos = nx.spring_layout(graph, seed=seed)
    nodes = sorted(str(node) for node in graph.nodes())
    color_map = _community_color_map(list(node_to_community.values()))
    node_colors = [
        color_map.get(node_to_community.get(node, -1), "#7f8c8d") for node in nodes
    ]

    nx.draw_networkx_edges(graph, pos, alpha=0.25, width=0.7, edge_color="#95a5a6")
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=nodes,
        node_size=36,
        node_color=node_colors,
        alpha=0.9,
        linewidths=0.25,
        edgecolors="#2c3e50",
    )
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def visualize_html(
    graph: nx.Graph,
    node_to_community: dict[str, int],
    out_html: Path,
    title: str,
    seed: int,
) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()

    if graph.number_of_nodes() == 0:
        fig.update_layout(title=title, template="plotly_white")
        fig.write_html(out_html, include_plotlyjs="cdn")
        return

    pos = nx.spring_layout(graph, seed=seed)
    nodes = sorted(str(node) for node in graph.nodes())

    edge_x: list[float] = []
    edge_y: list[float] = []
    for src, dst in graph.edges():
        x0, y0 = pos[str(src)]
        x1, y1 = pos[str(dst)]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            hoverinfo="none",
            line={"width": 0.8, "color": "#bdc3c7"},
            name="Edges",
        )
    )

    node_x: list[float] = []
    node_y: list[float] = []
    node_color: list[int] = []
    node_text: list[str] = []
    for node in nodes:
        x, y = pos[node]
        cid = int(node_to_community.get(node, -1))
        degree = int(graph.degree[node])
        node_x.append(x)
        node_y.append(y)
        node_color.append(cid)
        node_text.append(f"id={node}<br>community={cid}<br>degree={degree}")

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            text=node_text,
            hoverinfo="text",
            marker={
                "size": 9,
                "color": node_color,
                "colorscale": "Turbo",
                "line": {"width": 0.4, "color": "#2c3e50"},
                "colorbar": {"title": "Community"},
                "showscale": True,
            },
            name="Nodes",
        )
    )

    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    fig.write_html(out_html, include_plotlyjs="cdn")


def save_outputs(
    graph: nx.Graph,
    node_to_community: dict[str, int],
    communities: list[set[str]],
    modularity: float,
    out_dir: Path,
    community_method: str,
    recursion_stats: dict[str, Any] | None = None,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    communities_csv = out_dir / "louvain_communities.csv"
    summary_json = out_dir / "louvain_summary.json"
    png_path = out_dir / "louvain_community_graph.png"
    html_path = out_dir / "louvain_community_graph.html"

    rows = []
    for node in sorted(str(node) for node in graph.nodes()):
        rows.append(
            {
                "entity_id": node,
                "community_id": int(node_to_community.get(node, -1)),
                "degree": int(graph.degree[node]),
            }
        )
    pd.DataFrame(rows).to_csv(communities_csv, index=False)

    component_sizes = sorted(
        (len(comp) for comp in nx.connected_components(graph)),
        reverse=True,
    )
    summary = {
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
        "community_method": community_method,
        "connected_components": int(nx.number_connected_components(graph))
        if graph.number_of_nodes()
        else 0,
        "component_sizes_top10": [int(v) for v in component_sizes[:10]],
        "communities": int(len(communities)),
        "community_sizes_top10": [int(len(c)) for c in communities[:10]],
        "modularity": float(modularity),
        "recursive": recursion_stats,
        "files": {
            "communities_csv": str(communities_csv),
            "summary_json": str(summary_json),
            "graph_png": str(png_path),
            "graph_html": str(html_path),
        },
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    return {
        "communities_csv": str(communities_csv),
        "summary_json": str(summary_json),
        "graph_png": str(png_path),
        "graph_html": str(html_path),
    }


@click.command(
    help="Run Louvain community detection for matching predictions and visualize communities."
)
@click.option(
    "--predictions-csv",
    type=click.Path(path_type=Path, exists=True),
    default=Path("results/dirty_matching/cora_v7/predictions.csv"),
    show_default=True,
)
@click.option(
    "--edge-column",
    type=click.Choice(["pred", "label"]),
    default="pred",
    show_default=True,
    help="Which boolean column defines graph edges.",
)
@click.option("--resolution", type=float, default=1.0, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--max-community-size",
    type=int,
    default=10,
    show_default=True,
    help="Recursively split communities larger than this size.",
)
@click.option(
    "--max-recursion-depth",
    type=int,
    default=6,
    show_default=True,
    help="Stop recursive splitting when this depth is reached.",
)
@click.option(
    "--resolution-scale",
    type=float,
    default=1.25,
    show_default=True,
    help="Resolution multiplier applied at each recursive split.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory. Defaults to predictions parent folder.",
)
def main(
    predictions_csv: Path,
    edge_column: str,
    resolution: float,
    seed: int,
    max_community_size: int,
    max_recursion_depth: int,
    resolution_scale: float,
    output_dir: Path | None,
) -> None:
    df = load_predictions(predictions_csv)
    graph = build_graph(df, edge_column=edge_column)
    communities, node_to_community, modularity, recursion_stats = (
        detect_recursive_louvain_communities(
            graph,
            resolution=resolution,
            seed=seed,
            max_community_size=max_community_size,
            max_depth=max_recursion_depth,
            resolution_scale=resolution_scale,
        )
    )

    out_dir = output_dir if output_dir is not None else predictions_csv.parent
    title = (
        f"Recursive Louvain Communities ({predictions_csv.stem}, edge={edge_column})"
    )

    visualize_png(
        graph,
        node_to_community,
        out_png=out_dir / "louvain_community_graph.png",
        title=title,
        seed=seed,
    )
    visualize_html(
        graph,
        node_to_community,
        out_html=out_dir / "louvain_community_graph.html",
        title=title,
        seed=seed,
    )

    files = save_outputs(
        graph,
        node_to_community,
        communities,
        modularity,
        out_dir=out_dir,
        community_method="recursive_louvain",
        recursion_stats=recursion_stats,
    )

    print("Louvain run completed.")
    print(
        json.dumps(
            {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "communities": len(communities),
                "modularity": modularity,
                "recursive": recursion_stats,
                "files": files,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
