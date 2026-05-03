from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import pandas as pd

from src.dirty_matching.core.lookups import (
    build_entity_record_lookup,
    build_pair_prediction_lookup,
)
from src.dirty_matching.llm.local_resolvers import CommunityJointRefiner
from src.dirty_matching.refine.cache_paths import dirty_matching_stage_cache_dir


def _load_communities_from_stage1_csv(communities_csv: Path) -> dict[int, list[str]]:
    df = pd.read_csv(communities_csv)
    required_cols = {"entity_id", "community_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"communities.csv missing required columns: {sorted(missing)}")

    grouped: dict[int, list[str]] = {}
    for community_id, group in df.groupby("community_id"):
        cid = int(community_id)
        members = sorted({str(v) for v in group["entity_id"].tolist()})
        grouped[cid] = members
    return grouped


def _select_stage2_jobs(
    community_members: dict[int, list[str]],
    max_community_size_for_llm: int,
) -> list[tuple[int, list[str]]]:
    jobs: list[tuple[int, list[str]]] = []
    for community_id, members in sorted(community_members.items()):
        if len(members) < 2:
            continue
        if len(members) > max_community_size_for_llm:
            continue
        jobs.append((community_id, members))
    return jobs


def _community_pair_rows(predicted_df: pd.DataFrame, members: list[str]) -> pd.DataFrame:
    member_set = set(members)
    left = predicted_df["id_left"].astype(str)
    right = predicted_df["id_right"].astype(str)
    mask = left.isin(member_set) & right.isin(member_set)
    return predicted_df.loc[mask].copy()


def _print_community_snapshot(
    community_id: int,
    members: list[str],
    pair_rows: pd.DataFrame,
    record_lookup: dict[str, str],
    max_print_pairs: int,
) -> None:
    print("=" * 88)
    print(f"[Community {community_id}] members={len(members)}")
    print(f"Members: {members}")

    print(f"Pair rows inside community: {len(pair_rows)}")
    if len(pair_rows) > 0:
        cols = ["id_left", "id_right", "pred"]
        if "label" in pair_rows.columns:
            cols.append("label")
        preview = pair_rows[cols].head(max_print_pairs)
        print(preview.to_string(index=False))

    print("Related records:")
    for entity_id in members:
        record = str(record_lookup.get(entity_id, ""))
        print(f"- [{entity_id}] {record}")


def run(
    output_dir: Path,
    model_name: str,
    sample_size: int,
    seed: int,
    max_community_size_for_llm: int,
    max_retries: int,
    max_print_pairs: int,
) -> dict[str, Any]:
    predictions_csv = output_dir / "predictions.csv"
    communities_csv = output_dir / "refine" / "stage1_partition" / "communities.csv"

    if not predictions_csv.exists():
        raise FileNotFoundError(f"Missing file: {predictions_csv}")
    if not communities_csv.exists():
        raise FileNotFoundError(f"Missing file: {communities_csv}")

    predicted_df = pd.read_csv(predictions_csv)
    community_members = _load_communities_from_stage1_csv(communities_csv)
    jobs = _select_stage2_jobs(community_members, max_community_size_for_llm)
    if not jobs:
        raise RuntimeError("No stage2-eligible communities found with current size threshold.")

    rng = random.Random(seed)
    k = min(int(sample_size), len(jobs))
    sampled_jobs = rng.sample(jobs, k=k)

    record_lookup = build_entity_record_lookup(predicted_df)
    pair_lookup = build_pair_prediction_lookup(predicted_df)

    refiner = CommunityJointRefiner(
        model_name=model_name,
        max_retries=max_retries,
        cache_dir=dirty_matching_stage_cache_dir(model_name, "stage2"),
    )

    print(f"Total stage2-eligible communities: {len(jobs)}")
    print(f"Sampled communities (seed={seed}): {[cid for cid, _ in sampled_jobs]}")

    results: list[dict[str, Any]] = []
    for community_id, members in sampled_jobs:
        pair_rows = _community_pair_rows(predicted_df, members)
        _print_community_snapshot(
            community_id=community_id,
            members=members,
            pair_rows=pair_rows,
            record_lookup=record_lookup,
            max_print_pairs=max_print_pairs,
        )

        result = refiner.refine(
            community_id=community_id,
            entity_ids=members,
            record_lookup=record_lookup,
            pair_lookup=pair_lookup,
        )
        results.append(result)

        print("LLM output:")
        print(json.dumps(result, ensure_ascii=True, indent=2))

    payload = {
        "output_dir": str(output_dir),
        "model_name": model_name,
        "sample_size_requested": int(sample_size),
        "sample_size_actual": int(k),
        "seed": int(seed),
        "max_community_size_for_llm": int(max_community_size_for_llm),
        "eligible_communities": int(len(jobs)),
        "sampled_community_ids": [int(cid) for cid, _ in sampled_jobs],
        "results": results,
        "api_cost": float(refiner.cost),
    }

    out_json = (
        output_dir
        / "refine"
        / "stage2_local_resolving"
        / "sampled_local_resolve_results.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

    print("=" * 88)
    print(f"Saved sampled run result to: {out_json}")
    print(f"Total sampled LLM API cost: {payload['api_cost']:.6f}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample stage2-eligible communities from stage1 output, print records, "
            "and run CommunityJointRefiner on each sample."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dirty_matching/cora_new"),
        help="Pipeline output directory that contains predictions.csv and refine/stage1_partition/communities.csv.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt-5-mini",
        help="LLM model name used by CommunityJointRefiner.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="Number of communities to sample from stage2-eligible communities.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling communities.",
    )
    parser.add_argument(
        "--max-community-size-for-llm",
        type=int,
        default=50,
        help="Upper bound for stage2 local resolve eligibility.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for each LLM call.",
    )
    parser.add_argument(
        "--max-print-pairs",
        type=int,
        default=30,
        help="Max number of in-community pair rows to print for each sampled community.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.DEBUG)

    run(
        output_dir=args.output_dir,
        model_name=args.model_name,
        sample_size=args.sample_size,
        seed=args.seed,
        max_community_size_for_llm=args.max_community_size_for_llm,
        max_retries=args.max_retries,
        max_print_pairs=args.max_print_pairs,
    )


if __name__ == "__main__":
    main()