from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from src.dirty_matching.core.candidates import load_or_generate_candidates
from src.dirty_matching.core.graph import build_matching_graph, graph_statistics
from src.dirty_matching.core.prediction import predict_candidates
from src.dirty_matching.diagnostics.transitivity import detect_transitivity_violations
from src.dirty_matching.refine.stage1_partition import stage1_filter_and_split
from src.dirty_matching.refine.stage2_local_resolving import stage2_local_resolving
from src.dirty_matching.refine.stage3_golden_records import stage3_build_golden_records
from src.dirty_matching.refine.stage4_global_resolving import stage4_global_conflict_resolving
from src.dirty_matching.visualization.graph_html import visualize_graph_html
from src.dirty_matching.visualization.graph_png import visualize_graph_png


def _compute_pair_metrics(
    df: pd.DataFrame,
    pred_col: str,
    baseline_col: str | None = "pred",
) -> dict[str, Any]:
    pred_series = df[pred_col].astype(bool)
    baseline_series = (
        df[baseline_col].astype(bool)
        if baseline_col is not None and baseline_col in df.columns
        else None
    )

    metrics: dict[str, Any] = {
        "prediction_column": pred_col,
        "num_pairs": int(len(df)),
        "positive_predictions": int(pred_series.sum()),
        "positive_predictions_baseline": int(baseline_series.sum())
        if baseline_series is not None
        else None,
        "changed_pairs": int((pred_series != baseline_series).sum())
        if baseline_series is not None
        else None,
        "flipped_to_true": int((pred_series & (~baseline_series)).sum())
        if baseline_series is not None
        else None,
        "flipped_to_false": int(((~pred_series) & baseline_series).sum())
        if baseline_series is not None
        else None,
    }

    if "label" not in df.columns:
        return metrics

    labels = df["label"].astype(bool).tolist()
    preds = pred_series.tolist()
    report = classification_report(
        labels,
        preds,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, preds, labels=[False, True]).tolist()

    metrics["positive_labels"] = int(sum(labels))
    metrics["classification_report"] = report
    metrics["confusion_matrix"] = {
        "labels": [False, True],
        "matrix": matrix,
    }

    if baseline_series is not None:
        base_preds = baseline_series.tolist()
        base_report = classification_report(
            labels,
            base_preds,
            digits=4,
            output_dict=True,
            zero_division=0,
        )
        base_matrix = confusion_matrix(labels, base_preds, labels=[False, True]).tolist()
        metrics["classification_report_baseline"] = base_report
        metrics["confusion_matrix_baseline"] = {
            "labels": [False, True],
            "matrix": base_matrix,
        }

    return metrics


def run_refine_pipeline(
    graph: nx.Graph,
    predicted_df: pd.DataFrame,
    output_dir: Path,
    refine_model_name: str,
    community_detection_method: str,
    louvain_resolution: float,
    louvain_seed: int,
    max_community_size: int,
    max_recursion_depth: int,
    resolution_scale: float,
    local_resolve_max_size: int,
    local_resolve_workers: int,
    local_resolve_retries: int,
    golden_workers: int,
    golden_retries: int,
    golden_method: str,
    enable_global_arbitration: bool,
    global_resolve_workers: int,
    global_resolve_retries: int,
    max_global_conflicts: int,
    refine_max_stage: int,
) -> dict[str, Any]:
    refine_dir = output_dir / "refine"
    refine_dir.mkdir(parents=True, exist_ok=True)

    refine_max_stage = max(1, min(4, int(refine_max_stage)))

    refine_summary: dict[str, Any] = {
        "enabled": True,
        "model_name": refine_model_name,
        "max_stage_requested": int(refine_max_stage),
    }
    files: dict[str, Any] = {}
    api_cost_total = 0.0

    def finalize(completed_stage: int) -> dict[str, Any]:
        refine_summary["completed_stage"] = int(completed_stage)
        refine_summary["api_cost_total"] = float(api_cost_total)
        refine_summary["files"] = files

        summary_path = refine_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(refine_summary, f, ensure_ascii=True, indent=2)

        refine_summary["summary_json"] = str(summary_path)
        return refine_summary

    stage1 = stage1_filter_and_split(
        graph=graph,
        out_dir=refine_dir,
        resolution=louvain_resolution,
        seed=louvain_seed,
        max_community_size=max_community_size,
        max_recursion_depth=max_recursion_depth,
        resolution_scale=resolution_scale,
        method=community_detection_method,
    )

    stage1_dir = Path(stage1["files"]["summary_json"]).parent
    stage1_pair_df = predicted_df.copy()
    stage1_pair_df["community_id_left"] = stage1_pair_df["id_left"].map(
        lambda x: int(stage1["node_to_community"].get(str(x), -1))
    )
    stage1_pair_df["community_id_right"] = stage1_pair_df["id_right"].map(
        lambda x: int(stage1["node_to_community"].get(str(x), -1))
    )
    stage1_pair_df["pred_stage1"] = (
        stage1_pair_df["community_id_left"] == stage1_pair_df["community_id_right"]
    )

    stage1_predictions_csv = stage1_dir / "predictions_stage1.csv"
    stage1_pair_df.to_csv(stage1_predictions_csv, index=False)
    stage1_metrics = _compute_pair_metrics(
        stage1_pair_df,
        pred_col="pred_stage1",
        baseline_col="pred",
    )
    stage1_metrics_json = stage1_dir / "pair_metrics.json"
    with stage1_metrics_json.open("w", encoding="utf-8") as f:
        json.dump(stage1_metrics, f, ensure_ascii=True, indent=2)

    stage1["stats"]["pair_level_metrics"] = stage1_metrics
    stage1["files"]["predictions_stage1_csv"] = str(stage1_predictions_csv)
    stage1["files"]["pair_metrics_json"] = str(stage1_metrics_json)

    refine_summary["stage1"] = stage1["stats"]
    files["stage1"] = stage1["files"]
    if refine_max_stage == 1:
        return finalize(completed_stage=1)

    stage2 = stage2_local_resolving(
        predicted_df=predicted_df,
        communities=stage1["communities"],
        out_dir=refine_dir,
        model_name=refine_model_name,
        max_community_size_for_llm=local_resolve_max_size,
        max_workers=local_resolve_workers,
        max_retries=local_resolve_retries,
    )
    api_cost_total += float(stage2["stats"].get("api_cost", 0.0))

    stage2_dir = Path(stage2["files"]["summary_json"]).parent
    stage2_metrics = _compute_pair_metrics(
        stage2["predictions_local"],
        pred_col="pred_local",
        baseline_col="pred",
    )
    stage2_metrics_json = stage2_dir / "pair_metrics.json"
    with stage2_metrics_json.open("w", encoding="utf-8") as f:
        json.dump(stage2_metrics, f, ensure_ascii=True, indent=2)

    stage2["stats"]["pair_level_metrics"] = stage2_metrics
    stage2["files"]["pair_metrics_json"] = str(stage2_metrics_json)

    refine_summary["stage2"] = stage2["stats"]
    files["stage2"] = stage2["files"]
    if refine_max_stage == 2:
        return finalize(completed_stage=2)

    stage3 = stage3_build_golden_records(
        predicted_df=predicted_df,
        local_clusters=stage2["local_clusters"],
        local_cluster_to_community=stage2["local_cluster_to_community"],
        out_dir=refine_dir,
        model_name=refine_model_name,
        max_workers=golden_workers,
        max_retries=golden_retries,
        golden_method=golden_method,
    )
    api_cost_total += float(stage3["stats"].get("api_cost", 0.0))

    refine_summary["stage3"] = stage3["stats"]
    files["stage3"] = stage3["files"]
    if refine_max_stage == 3:
        return finalize(completed_stage=3)

    stage4 = stage4_global_conflict_resolving(
        predicted_df=predicted_df,
        golden_records=stage3["golden_records"],
        entity_to_golden=stage3["entity_to_golden"],
        out_dir=refine_dir,
        model_name=refine_model_name,
        enable_global_arbitration=enable_global_arbitration,
        max_workers=global_resolve_workers,
        max_retries=global_resolve_retries,
        max_conflicts=max_global_conflicts,
    )
    api_cost_total += float(stage4["stats"].get("api_cost", 0.0))

    refine_summary["stage4"] = stage4["stats"]
    files["stage4"] = stage4["files"]
    return finalize(completed_stage=4)


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
        cluster_layout_k=None,
        cluster_layout_iterations=350,
        cluster_layout_spread=None,
        cluster_layout_norm_quantile=0.9,
        enable_refine=False,
        refine_model_name="gpt-4o",
        community_detection_method="louvain",
        louvain_resolution=1.0,
        louvain_seed=42,
        max_community_size=10,
        max_recursion_depth=6,
        resolution_scale=1.25,
        local_resolve_max_size=10,
        local_resolve_workers=8,
        local_resolve_retries=3,
        golden_workers=8,
        golden_retries=3,
        golden_method="medoid",
        enable_global_arbitration=True,
        global_resolve_workers=8,
        global_resolve_retries=3,
        max_global_conflicts=200,
        refine_max_stage=4,
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
    cluster_layout_k: float | None,
    cluster_layout_iterations: int,
    cluster_layout_spread: float | None,
    cluster_layout_norm_quantile: float,
    enable_refine: bool,
    refine_model_name: str,
    community_detection_method: str,
    louvain_resolution: float,
    louvain_seed: int,
    max_community_size: int,
    max_recursion_depth: int,
    resolution_scale: float,
    local_resolve_max_size: int,
    local_resolve_workers: int,
    local_resolve_retries: int,
    golden_workers: int,
    golden_retries: int,
    golden_method: str,
    enable_global_arbitration: bool,
    global_resolve_workers: int,
    global_resolve_retries: int,
    max_global_conflicts: int,
    refine_max_stage: int,
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

    if enable_refine:
        refine_summary = run_refine_pipeline(
            graph=graph,
            predicted_df=predicted_df,
            output_dir=output_dir,
            refine_model_name=refine_model_name,
            community_detection_method=community_detection_method,
            louvain_resolution=louvain_resolution,
            louvain_seed=louvain_seed,
            max_community_size=max_community_size,
            max_recursion_depth=max_recursion_depth,
            resolution_scale=resolution_scale,
            local_resolve_max_size=local_resolve_max_size,
            local_resolve_workers=local_resolve_workers,
            local_resolve_retries=local_resolve_retries,
            golden_workers=golden_workers,
            golden_retries=golden_retries,
            golden_method=golden_method,
            enable_global_arbitration=enable_global_arbitration,
            global_resolve_workers=global_resolve_workers,
            global_resolve_retries=global_resolve_retries,
            max_global_conflicts=max_global_conflicts,
            refine_max_stage=refine_max_stage,
        )
        summary["refine_pipeline"] = refine_summary
        summary["files"]["refine_summary_json"] = str(refine_summary["summary_json"])
    else:
        summary["refine_pipeline"] = {"enabled": False}

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    print("Run completed.")
    print(json.dumps(summary["graph"], ensure_ascii=True, indent=2))
    print(json.dumps(summary["transitivity"], ensure_ascii=True, indent=2))
    if summary["refine_pipeline"].get("enabled"):
        completed_stage = int(summary["refine_pipeline"].get("completed_stage", 0))
        if completed_stage > 0:
            stage_key = f"stage{completed_stage}"
            print(json.dumps(summary["refine_pipeline"].get(stage_key, {}), ensure_ascii=True, indent=2))
    print(f"API cost: {metrics['api_cost']:.4f}")
    print(f"Outputs written to: {output_dir}")

    return summary
