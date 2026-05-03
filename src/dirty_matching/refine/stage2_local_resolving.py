from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.contrib.concurrent import thread_map

from src.dirty_matching.core.lookups import (
    build_entity_record_lookup,
    build_pair_prediction_lookup,
)
from src.dirty_matching.llm.local_resolvers import CommunityJointRefiner
from src.dirty_matching.refine.cache_paths import dirty_matching_stage_cache_dir


def stage2_local_resolving(
    predicted_df: pd.DataFrame,
    communities: list[set[str]],
    out_dir: Path,
    model_name: str,
    max_community_size_for_llm: int,
    max_workers: int,
    max_retries: int,
) -> dict[str, Any]:
    stage_dir = out_dir / "stage2_local_resolving"
    stage_dir.mkdir(parents=True, exist_ok=True)

    record_lookup = build_entity_record_lookup(predicted_df)
    pair_lookup = build_pair_prediction_lookup(predicted_df)

    refiner = CommunityJointRefiner(
        model_name=model_name,
        max_retries=max_retries,
        cache_dir=dirty_matching_stage_cache_dir(model_name, "stage2"),
    )

    jobs: list[tuple[int, list[str]]] = []
    for community_id, community in enumerate(communities):
        members = sorted(str(node) for node in community)
        if len(members) < 2:
            continue
        if len(members) > max_community_size_for_llm:
            continue
        jobs.append((community_id, members))

    def run_job(job: tuple[int, list[str]]) -> dict[str, Any]:
        community_id, members = job
        return refiner.refine(
            community_id=community_id,
            entity_ids=members,
            record_lookup=record_lookup,
            pair_lookup=pair_lookup,
        )

    llm_results = (
        thread_map(run_job, jobs, max_workers=max(1, int(max_workers))) if jobs else []
    )
    llm_result_lookup = {int(result["community_id"]): result for result in llm_results}

    local_clusters: dict[str, list[str]] = {}
    local_cluster_to_community: dict[str, int] = {}
    entity_to_local_cluster: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    fallback_large_communities = 0

    for community_id, community in enumerate(communities):
        members = sorted(str(node) for node in community)
        if not members:
            continue

        result = llm_result_lookup.get(community_id)
        status = "no_llm_single_cluster"
        clusters = [members]
        if result is not None:
            status = str(result.get("status", "ok"))
            clusters = [sorted(str(v) for v in cluster) for cluster in result["clusters"]]
        elif len(members) > max_community_size_for_llm:
            fallback_large_communities += 1

        for local_idx, cluster_members in enumerate(clusters):
            local_cluster_id = f"c{community_id}_l{local_idx}"
            local_clusters[local_cluster_id] = cluster_members
            local_cluster_to_community[local_cluster_id] = int(community_id)
            for entity_id in cluster_members:
                entity_to_local_cluster[entity_id] = local_cluster_id
            rows.append(
                {
                    "local_cluster_id": local_cluster_id,
                    "community_id": int(community_id),
                    "size": int(len(cluster_members)),
                    "members": json.dumps(cluster_members, ensure_ascii=True),
                    "status": status,
                }
            )

    for node in sorted(set(predicted_df["id_left"]).union(set(predicted_df["id_right"]))):
        node_id = str(node)
        if node_id in entity_to_local_cluster:
            continue
        local_cluster_id = f"solo_{node_id}"
        local_clusters[local_cluster_id] = [node_id]
        local_cluster_to_community[local_cluster_id] = -1
        entity_to_local_cluster[node_id] = local_cluster_id
        rows.append(
            {
                "local_cluster_id": local_cluster_id,
                "community_id": -1,
                "size": 1,
                "members": json.dumps([node_id], ensure_ascii=True),
                "status": "solo_completion",
            }
        )

    local_clusters_df = pd.DataFrame(rows)
    local_clusters_csv = stage_dir / "local_clusters.csv"
    local_clusters_df.to_csv(local_clusters_csv, index=False)

    entity_rows = [
        {
            "entity_id": entity_id,
            "local_cluster_id": cluster_id,
            "community_id": int(local_cluster_to_community.get(cluster_id, -1)),
        }
        for entity_id, cluster_id in sorted(entity_to_local_cluster.items())
    ]
    entity_assign_csv = stage_dir / "entity_local_cluster.csv"
    pd.DataFrame(entity_rows).to_csv(entity_assign_csv, index=False)

    predictions_local = predicted_df.copy()
    predictions_local["local_cluster_id_left"] = predictions_local["id_left"].map(
        lambda x: entity_to_local_cluster.get(str(x), "")
    )
    predictions_local["local_cluster_id_right"] = predictions_local["id_right"].map(
        lambda x: entity_to_local_cluster.get(str(x), "")
    )
    predictions_local["pred_local"] = (
        predictions_local["local_cluster_id_left"]
        == predictions_local["local_cluster_id_right"]
    )
    predictions_local_csv = stage_dir / "predictions_local.csv"
    predictions_local.to_csv(predictions_local_csv, index=False)

    llm_status_counts = (
        pd.Series([result.get("status", "") for result in llm_results])
        .value_counts()
        .to_dict()
        if llm_results
        else {}
    )
    stats = {
        "communities_total": int(len(communities)),
        "communities_sent_to_llm": int(len(jobs)),
        "communities_over_size_limit": int(fallback_large_communities),
        "local_clusters": int(len(local_clusters)),
        "entities": int(len(entity_to_local_cluster)),
        "llm_status_counts": llm_status_counts,
        "api_cost": float(refiner.cost),
    }
    summary_json = stage_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=True, indent=2)

    return {
        "local_clusters": local_clusters,
        "local_cluster_to_community": local_cluster_to_community,
        "entity_to_local_cluster": entity_to_local_cluster,
        "predictions_local": predictions_local,
        "stats": stats,
        "files": {
            "local_clusters_csv": str(local_clusters_csv),
            "entity_assign_csv": str(entity_assign_csv),
            "predictions_local_csv": str(predictions_local_csv),
            "summary_json": str(summary_json),
        },
    }
