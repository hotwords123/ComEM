from __future__ import annotations

import html
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go
from matplotlib import pyplot as plt


def _node_color_by_community(graph: nx.Graph, node: str) -> str:
    community = graph.nodes[node].get("community_id", -1)
    if community is None or int(community) < 0:
        return "#7f8c8d"

    palette = [
        "#1f77b4",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#17becf",
        "#e377c2",
        "#8c564b",
        "#bcbd22",
        "#d62728",
        "#7f7f7f",
    ]
    return palette[int(community) % len(palette)]


def visualize_golden_graph_png(graph: nx.Graph, out_png: Path, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 10))
    if graph.number_of_nodes() == 0:
        plt.title(f"{title} (empty graph)")
        plt.axis("off")
        plt.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close()
        return

    pos = nx.spring_layout(graph, seed=42, weight="weight")

    widths = []
    for left, right in graph.edges():
        weight = float(graph[left][right].get("weight", 1.0))
        widths.append(min(4.0, 0.7 + 0.45 * weight))

    nx.draw_networkx_edges(graph, pos, alpha=0.35, width=widths, edge_color="#5f6a6a")

    nodes = sorted(str(node) for node in graph.nodes())
    sizes = [
        min(1200.0, 220.0 + 45.0 * float(graph.nodes[node].get("member_size", 1)))
        for node in nodes
    ]
    colors = [_node_color_by_community(graph, node) for node in nodes]

    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=nodes,
        node_size=sizes,
        node_color=colors,
        alpha=0.9,
        linewidths=0.5,
        edgecolors="#2c3e50",
    )

    nx.draw_networkx_labels(graph, pos, labels={n: n for n in nodes}, font_size=8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def visualize_golden_graph_html(graph: nx.Graph, out_html: Path, title: str) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)

    fig = go.Figure()
    if graph.number_of_nodes() == 0:
        fig.update_layout(title=title, template="plotly_white")
        fig.write_html(out_html, include_plotlyjs="cdn")
        return

    pos = nx.spring_layout(graph, seed=42, weight="weight")

    edge_x: list[float] = []
    edge_y: list[float] = []
    edge_text_x: list[float] = []
    edge_text_y: list[float] = []
    edge_text: list[str] = []

    for left, right in graph.edges():
        x0, y0 = pos[str(left)]
        x1, y1 = pos[str(right)]
        weight = int(graph[str(left)][str(right)].get("weight", 1))
        support_pairs = graph[str(left)][str(right)].get("support_pairs", [])

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        edge_text_x.append((x0 + x1) / 2.0)
        edge_text_y.append((y0 + y1) / 2.0)
        edge_text.append(
            "<br>".join(
                [
                    f"golden_left={html.escape(str(left))}",
                    f"golden_right={html.escape(str(right))}",
                    f"support_edges={weight}",
                    f"example_pairs={html.escape(str(support_pairs))}",
                ]
            )
        )

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": 1.0, "color": "#7f8c8d"},
            hoverinfo="none",
            name=f"Edges ({graph.number_of_edges()})",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=edge_text_x,
            y=edge_text_y,
            mode="markers",
            marker={"size": 5, "color": "rgba(127,140,141,0.55)"},
            text=edge_text,
            hoverinfo="text",
            name="Edge details",
            showlegend=False,
        )
    )

    node_x: list[float] = []
    node_y: list[float] = []
    node_text: list[str] = []
    node_color: list[str] = []
    node_size: list[float] = []

    for node in sorted(str(n) for n in graph.nodes()):
        x, y = pos[node]
        community_id = int(graph.nodes[node].get("community_id", -1))
        member_size = int(graph.nodes[node].get("member_size", 1))
        record_preview = str(graph.nodes[node].get("record_preview", ""))

        node_x.append(x)
        node_y.append(y)
        node_color.append(_node_color_by_community(graph, node))
        node_size.append(min(28.0, 11.0 + 1.2 * member_size))
        node_text.append(
            "<br>".join(
                [
                    f"golden_id={html.escape(node)}",
                    f"community_id={community_id}",
                    f"member_size={member_size}",
                    f"record={html.escape(record_preview)}",
                ]
            )
        )

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[str(n) for n in sorted(str(v) for v in graph.nodes())],
            textposition="top center",
            hoverinfo="text",
            hovertext=node_text,
            marker={
                "size": node_size,
                "color": node_color,
                "line": {"width": 0.8, "color": "#2c3e50"},
            },
            name=f"Golden nodes ({graph.number_of_nodes()})",
            showlegend=True,
        )
    )

    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 60, "b": 40},
        xaxis={"visible": False},
        yaxis={"visible": False},
        showlegend=True,
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.05},
    )
    fig.write_html(out_html, include_plotlyjs="cdn")
