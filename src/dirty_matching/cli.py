from __future__ import annotations

from pathlib import Path

import click

from src.dirty_matching.pipeline import run_pipeline


@click.command(
    help="Run dirty ER matching evaluation, build a match graph, and audit transitivity violations."
)
@click.option(
    "--dataset-name",
    type=click.Choice(["cora", "wdc", "cddb", "musicbrainz"], case_sensitive=False),
    default="cora",
    show_default=True,
)
@click.option(
    "--reader-root",
    type=click.Path(path_type=Path),
    default=None,
    show_default=False,
)
@click.option("--candidates-csv", type=click.Path(path_type=Path), default=None)
@click.option("--model-name", type=str, default="gpt-4o-mini", show_default=True)
@click.option("--topk", type=int, default=20, show_default=True)
@click.option("--max-workers", type=int, default=16, show_default=True)
@click.option(
    "--violation-mode",
    type=click.Choice(["strict_negative", "include_missing"]),
    default="strict_negative",
    show_default=True,
)
@click.option("--sample-frac", type=float, default=None)
@click.option("--sample-n", type=int, default=200, show_default=True)
@click.option(
    "--sample-cluster-n",
    type=int,
    default=None,
    help="Sample N clusters (take all records in each sampled cluster).",
)
@click.option("--sample-seed", type=int, default=42, show_default=True)
@click.option(
    "--cluster-layout-k",
    type=float,
    default=None,
    help="Spring-layout ideal distance for GT-cluster centers.",
)
@click.option(
    "--cluster-layout-iterations",
    type=int,
    default=350,
    show_default=True,
    help="Iterations for GT-cluster center spring-layout.",
)
@click.option(
    "--cluster-layout-spread",
    type=float,
    default=None,
    help="Global spread scale for cluster centers after normalization.",
)
@click.option(
    "--cluster-layout-norm-quantile",
    type=float,
    default=0.9,
    show_default=True,
    help="Quantile used to robustly normalize cluster-center distances.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option(
    "--force-rebuild-index",
    is_flag=True,
    default=False,
    help="Force rebuilding SparseRetriever index before searching.",
)
@click.option(
    "--enable-refine",
    is_flag=True,
    default=False,
    help="Enable four-stage refine ER pipeline (Louvain + local/global LLM resolving).",
)
@click.option("--refine-model-name", type=str, default="gpt-4o", show_default=True)
@click.option(
    "--community-detection-method",
    type=click.Choice(["louvain", "greedy", "label_prop", "k_clique", "spectral"]),
    default="louvain",
    show_default=True,
    help="Community detection algorithm for stage1 partitioning.",
)
@click.option("--louvain-resolution", type=float, default=1.0, show_default=True)
@click.option("--louvain-seed", type=int, default=42, show_default=True)
@click.option("--max-community-size", type=int, default=10, show_default=True)
@click.option("--max-recursion-depth", type=int, default=6, show_default=True)
@click.option(
    "--resolution-scale",
    type=float,
    default=1.25,
    show_default=True,
    help="Recursive Louvain resolution multiplier (>1.0).",
)
@click.option(
    "--local-resolve-max-size",
    type=int,
    default=10,
    show_default=True,
    help="Only communities with size <= this threshold are sent to local LLM resolving.",
)
@click.option("--local-resolve-workers", type=int, default=8, show_default=True)
@click.option("--local-resolve-retries", type=int, default=3, show_default=True)
@click.option("--golden-workers", type=int, default=8, show_default=True)
@click.option("--golden-retries", type=int, default=3, show_default=True)
@click.option(
    "--golden-method",
    type=click.Choice(["medoid", "llm"]),
    default="medoid",
    show_default=True,
    help="Stage3 golden-record strategy: graph 1-medoid or LLM synthesis.",
)
@click.option(
    "--enable-global-arbitration/--disable-global-arbitration",
    default=True,
    show_default=True,
    help="Enable LLM arbitration for macro conflicts in stage4.",
)
@click.option("--global-resolve-workers", type=int, default=8, show_default=True)
@click.option("--global-resolve-retries", type=int, default=3, show_default=True)
@click.option(
    "--max-global-conflicts",
    type=int,
    default=200,
    show_default=True,
    help="Maximum number of pair/triangle conflicts to send to stage4 arbitration.",
)
@click.option(
    "--refine-max-stage",
    type=click.IntRange(1, 4),
    default=4,
    show_default=True,
    help="Run refine pipeline only up to this stage (1-4).",
)
def main(*args, **kwargs) -> None:
    run_pipeline(*args, **kwargs)


if __name__ == "__main__":
    main()
