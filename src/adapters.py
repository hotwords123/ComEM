from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class EntityTableAdapter:
    """Load a raw dataset into a unified dirty-ER entity table."""

    dataset_name: str
    raw_path: Path

    def load_entity_table(self) -> pd.DataFrame:
        raise NotImplementedError


class CoraAdapter(EntityTableAdapter):
    """Adapter for the Cora dirty ER dataset."""

    DEFAULT_CORE_FIELDS = ("title", "author", "venue", "year")

    def __init__(self, raw_path: Path):
        super().__init__(dataset_name="cora", raw_path=raw_path)

    def load_entity_table(self) -> pd.DataFrame:
        raw = pd.read_csv(self.raw_path / "cora.csv", sep="|", dtype=str)
        raw = raw.fillna("")

        if "Entity Id" not in raw.columns:
            raise ValueError("cora.csv is missing the Entity Id column")

        raw = raw.rename(columns={"Entity Id": "entity_id"})
        if "" in raw.columns:
            raw = raw.drop(columns=[""])

        text_columns = [c for c in self.DEFAULT_CORE_FIELDS if c in raw.columns]

        if not text_columns:
            text_columns = [column for column in raw.columns if column != "entity_id"]

        print(f"Using text fields for CORA: {text_columns}")

        raw["record"] = (
            raw[text_columns]
            .astype(str)
            .agg(" ".join, axis=1)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        raw["cluster_id"] = self._build_cluster_ids(self.raw_path / "cora_gt.csv", raw)

        return raw[["entity_id", "cluster_id", "record"]].copy()

    def _build_cluster_ids(self, gt_path: Path, raw: pd.DataFrame) -> pd.Series:
        gt = pd.read_csv(gt_path, sep="|", header=None, names=["left", "right"], dtype=str)
        gt = gt.dropna(how="any")

        parent = {entity_id: entity_id for entity_id in raw["entity_id"].astype(str)}

        def find(entity_id: str) -> str:
            parent.setdefault(entity_id, entity_id)
            while parent[entity_id] != entity_id:
                parent[entity_id] = parent[parent[entity_id]]
                entity_id = parent[entity_id]
            return entity_id

        def union(left: str, right: str) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for left, right in gt.itertuples(index=False, name=None):
            union(str(left), str(right))

        root_to_cluster: dict[str, int] = {}
        cluster_ids: list[int] = []
        for entity_id in raw["entity_id"].astype(str):
            root = find(entity_id)
            if root not in root_to_cluster:
                root_to_cluster[root] = len(root_to_cluster)
            cluster_ids.append(root_to_cluster[root])

        return pd.Series(cluster_ids, index=raw.index, name="cluster_id")


class WdcProductsAdapter(EntityTableAdapter):
    """Adapter for the WDC Products multi-class dataset (JSON Lines)."""

    DEFAULT_CORE_FIELDS = ("brand", "title", "description", "price", "priceCurrency")
    DEFAULT_FILENAME = "wdcproductsmulti80cc20rnd100un_gs.json"

    def __init__(self, raw_path: Path, filename: str | None = None):
        super().__init__(dataset_name="wdc", raw_path=raw_path)
        self._filename = filename or self.DEFAULT_FILENAME

    def load_entity_table(self) -> pd.DataFrame:
        input_path = self.raw_path / self._filename
        if not input_path.exists():
            raise FileNotFoundError(
                f"WDC input file not found: {input_path}. "
                f"Expected a JSONL file like '{self.DEFAULT_FILENAME}'."
            )

        raw = pd.read_json(input_path, lines=True)
        raw = raw.fillna("")

        required_columns = {"id", "cluster_id"}
        missing = required_columns - set(raw.columns)
        if missing:
            raise ValueError(f"{input_path.name} is missing columns: {sorted(missing)}")

        raw = raw.rename(columns={"id": "entity_id"})
        raw["entity_id"] = raw["entity_id"].astype(str)

        text_columns = [c for c in self.DEFAULT_CORE_FIELDS if c in raw.columns]
        if not text_columns:
            text_columns = [c for c in raw.columns if c not in {"entity_id", "cluster_id"}]

        print(f"Using text fields for WDC: {text_columns}")

        raw["record"] = (
            raw[text_columns]
            .astype(str)
            .agg(" ".join, axis=1)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        if "cluster_id" in raw.columns:
            raw["cluster_id"] = pd.to_numeric(raw["cluster_id"], errors="coerce")
        if raw["cluster_id"].isna().any():
            raise ValueError(
                f"{input_path.name} contains missing/non-numeric cluster_id values"
            )
        raw["cluster_id"] = raw["cluster_id"].astype(int)

        return raw[["entity_id", "cluster_id", "record"]].copy()


def get_entity_table_adapter(
    dataset_name: str,
    raw_path: Path,
) -> EntityTableAdapter:
    if dataset_name == "cora":
        return CoraAdapter(raw_path)
    if dataset_name in {"wdc", "wdc-products", "wdc_products"}:
        return WdcProductsAdapter(raw_path)
    raise ValueError(f"Unsupported dirty ER dataset: {dataset_name}")