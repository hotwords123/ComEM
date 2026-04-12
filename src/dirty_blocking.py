from __future__ import annotations

from math import comb
from pathlib import Path

import nltk
import pandas as pd
from retriv import SparseRetriever

from src.adapters import get_entity_table_adapter

nltk.download = lambda *args, **kwargs: None


def generate_docs(df: pd.DataFrame):
    for _, row in df.iterrows():
        yield {
            "id": str(row["entity_id"]),
            "text": str(row["record"]),
        }


def compute_entity_statistics(entity_table: pd.DataFrame) -> dict[str, object]:
    cluster_sizes = entity_table.groupby("cluster_id").size()
    matches = sum(comb(int(size), 2) for size in cluster_sizes if int(size) >= 2)
    quantile_levels = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
    quantiles = {
        f"q{int(level * 100):02d}": float(cluster_sizes.quantile(level))
        for level in quantile_levels
    }

    return {
        "entities": int(len(entity_table)),
        "clusters": int(cluster_sizes.size),
        "matches": int(matches),
        "cluster_size_mean": float(cluster_sizes.mean()),
        "cluster_size_quantiles": quantiles,
    }


def build_candidates(
    entity_table: pd.DataFrame,
    topk: int = 20,
    index_name: str = "dirty-index",
    force_rebuild_index: bool = False,
):
    entity_stats = compute_entity_statistics(entity_table)

    if force_rebuild_index:
        retriever = SparseRetriever(index_name=index_name)
        retriever = retriever.index(generate_docs(entity_table), show_progress=True)
    else:
        try:
            retriever = SparseRetriever.load(index_name)
        except FileNotFoundError:
            retriever = SparseRetriever(index_name=index_name)
            retriever = retriever.index(generate_docs(entity_table), show_progress=True)

    queries = list(generate_docs(entity_table))
    candidates = retriever.bsearch(queries, show_progress=True, cutoff=topk + 1)

    candidate_pairs = []
    seen_pairs: set[tuple[str, str]] = set()
    for query_id, scored_candidates in candidates.items():
        for candidate_id in sorted(
            scored_candidates, key=scored_candidates.get, reverse=True
        ):
            if str(query_id) == str(candidate_id):
                continue
            left_id, right_id = sorted((str(query_id), str(candidate_id)))
            pair = (left_id, right_id)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            candidate_pairs.append(pair)

    candidates = pd.DataFrame(candidate_pairs, columns=["id_left", "id_right"])

    entity_lookup = entity_table.rename(columns={"entity_id": "id"})
    left = entity_lookup.rename(
        columns={
            "id": "id_left",
            "record": "record_left",
            "cluster_id": "cluster_id_left",
        }
    )
    right = entity_lookup.rename(
        columns={
            "id": "id_right",
            "record": "record_right",
            "cluster_id": "cluster_id_right",
        }
    )

    candidates = candidates.merge(
        left[["id_left", "record_left", "cluster_id_left"]],
        on="id_left",
        how="left",
    )
    candidates = candidates.merge(
        right[["id_right", "record_right", "cluster_id_right"]],
        on="id_right",
        how="left",
    )
    candidates["label"] = candidates["cluster_id_left"] == candidates["cluster_id_right"]

    candidate_stats = {
        "candidate_pairs": int(len(candidates)),
        "positive_candidates": int(candidates["label"].sum()),
        "recall": (
            float(candidates["label"].sum()) / entity_stats["matches"]
            if entity_stats["matches"]
            else 0.0
        ),
    }

    print("Dataset statistics:")
    print(f"  #entities: {entity_stats['entities']}")
    print(f"  #clusters: {entity_stats['clusters']}")
    print(f"  #matches: {entity_stats['matches']}")
    print(f"  cluster_size_mean: {entity_stats['cluster_size_mean']:.2f}")
    print("  cluster_size_quantiles:")
    for key, value in entity_stats["cluster_size_quantiles"].items():
        print(f"    {key}: {value:.2f}")
    print("Blocking quality:")
    print(f"  #candidate_pairs: {candidate_stats['candidate_pairs']}")
    print(f"  #positive_candidates: {candidate_stats['positive_candidates']}")
    print(f"  Recall@{topk}: {candidate_stats['recall']:.4f}")

    return candidates


def sample_entity_table(
    entity_table: pd.DataFrame,
    sample_frac: float | None = None,
    sample_n: int | None = None,
    sample_seed: int = 42,
) -> pd.DataFrame:
    if sample_frac is not None and sample_n is not None:
        raise ValueError("Use either sample_frac or sample_n, not both")

    if sample_frac is not None:
        if not 0 < sample_frac <= 1:
            raise ValueError("sample_frac must be in (0, 1]")
        sampled = entity_table.sample(frac=sample_frac, random_state=sample_seed)
    elif sample_n is not None:
        if sample_n <= 0:
            raise ValueError("sample_n must be > 0")
        sample_n = min(sample_n, len(entity_table))
        sampled = entity_table.sample(n=sample_n, random_state=sample_seed)
    else:
        sampled = entity_table

    sampled = sampled.reset_index(drop=True)
    if len(sampled) < 2:
        raise ValueError("Sampled subset must contain at least 2 entities")

    return sampled


def run_dirty_pipeline(
    dataset_name: str,
    reader_root: Path,
    topk: int = 20,
    output_path: Path | None = None,
    force_rebuild_index: bool = False,
    sample_frac: float | None = None,
    sample_n: int | None = None,
    sample_seed: int = 42,
):
    adapter = get_entity_table_adapter(dataset_name, reader_root)
    full_entity_table = adapter.load_entity_table()
    entity_table = sample_entity_table(
        full_entity_table,
        sample_frac=sample_frac,
        sample_n=sample_n,
        sample_seed=sample_seed,
    )
    if len(entity_table) != len(full_entity_table):
        print(
            "Sampling subset: "
            f"{len(entity_table)}/{len(full_entity_table)} entities "
            f"(seed={sample_seed})"
        )

    candidates = build_candidates(
        entity_table,
        topk=topk,
        index_name=f"{dataset_name}-dirty-index",
        force_rebuild_index=force_rebuild_index,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(output_path, index=False)
        print(f"Saved {len(candidates)} candidate pairs to {output_path}")

    return candidates


if __name__ == "__main__":
    run_dirty_pipeline(
        dataset_name="cora",
        reader_root=Path("data/pyJedAI/data/der/cora"),
        topk=20,
        output_path=Path("data/llm4em/dirty/cora.csv"),
        force_rebuild_index=True,
        sample_n=200,
        sample_seed=42,
    )
