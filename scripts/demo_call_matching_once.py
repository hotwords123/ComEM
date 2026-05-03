from __future__ import annotations

import argparse
import logging
import random
from typing import Any

import pandas as pd

from src.dirty_matching.core.candidates import normalize_candidates
from src.matching import Matching


def build_demo_candidates(seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)

    left_record = "name=Apple iPhone 14 Pro; brand=Apple; color=black; storage=128GB"

    right_pool = [
        "name=Apple iPhone 14 Pro 128GB; brand=Apple; color=Black",
        "name=Apple iPhone 14 Pro Max 128GB; brand=Apple; color=Black",
        "name=Samsung Galaxy S23 128GB; brand=Samsung; color=Black",
        "name=Apple iPhone 13 Pro 128GB; brand=Apple; color=Black",
        "name=Apple iPhone 14 Pro; brand=Apple; color=black; storage=128GB",
    ]
    rng.shuffle(right_pool)

    rows: list[dict[str, Any]] = []
    for idx, right in enumerate(right_pool, start=1):
        rows.append(
            {
                "id_left": "L1",
                "id_right": f"R{idx}",
                "record_left": left_record,
                "record_right": right,
                "label": "iphone 14 pro" in right.lower() and "max" not in right.lower(),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate demo candidates and call Matching once with prediction.py-compatible interface."
    )
    parser.add_argument("--model-name", type=str, default="gpt-4o-mini")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.DEBUG)

    candidates = build_demo_candidates(seed=args.seed)
    df = normalize_candidates(candidates)

    matcher = Matching(model_name=args.model_name)

    grouped = list(df.groupby("id_left", sort=False))
    _, group = grouped[0]
    instance = {
        "anchor": group["record_left"].iloc[0],
        "candidates": group["record_right"].tolist(),
        "row_indexes": group.index.tolist(),
    }

    # Same interface and parameters as prediction.py: matcher(instance)
    preds = matcher(instance)

    print("Demo candidates:")
    print(df[["id_left", "id_right", "record_right", "label"]])
    print("\nPredictions from a single matcher(instance) call:")
    print(pd.DataFrame({"row_index": instance["row_indexes"], "pred": preds}))
    print(f"\nAPI cost: {matcher.cost:.6f}")


if __name__ == "__main__":
    main()
