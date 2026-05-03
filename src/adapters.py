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


class CddbAdapter(EntityTableAdapter):
    """Adapter for the CDDB dirty ER dataset (CSV + pairwise ground truth)."""

    DEFAULT_CORE_FIELDS = ("artist", "category", "genre", "title", "tracks", "year")
    DEFAULT_FILENAME = "cddb.csv"
    DEFAULT_GT_FILENAME = "gt.csv"

    def __init__(self, raw_path: Path):
        super().__init__(dataset_name="cddb", raw_path=raw_path)

    def load_entity_table(self) -> pd.DataFrame:
        input_path = self.raw_path / self.DEFAULT_FILENAME
        if not input_path.exists():
            raise FileNotFoundError(f"CDDB input file not found: {input_path}")

        gt_path = self.raw_path / self.DEFAULT_GT_FILENAME
        gt = self._load_ground_truth_pairs(gt_path)
        gt_ids = set(gt["id1"].astype(str).str.strip().tolist()) | set(
            gt["id2"].astype(str).str.strip().tolist()
        )
        gt_ids.discard("")

        raw = pd.read_csv(input_path, dtype=str)
        raw = raw.fillna("")

        if "id" not in raw.columns:
            raise ValueError(f"{input_path.name} is missing the id column")

        raw = raw.rename(columns={"id": "entity_id"})
        raw["entity_id"] = raw["entity_id"].astype(str)

        before = int(len(raw))
        raw = raw[raw["entity_id"].astype(str).isin(gt_ids)].reset_index(drop=True)
        after = int(len(raw))
        if after == 0:
            raise ValueError(
                "CDDB filtering removed all records. "
                f"No entity_id values from {gt_path} were found in {input_path.name}."
            )
        if after != before:
            print(
                f"CDDB pre-filter by GT ids: kept {after}/{before} records "
                f"({after / max(1, before):.1%})"
            )

        text_columns = [c for c in self.DEFAULT_CORE_FIELDS if c in raw.columns]
        if not text_columns:
            text_columns = [c for c in raw.columns if c != "entity_id"]

        print(f"Using text fields for CDDB: {text_columns}")

        raw["record"] = (
            raw[text_columns]
            .astype(str)
            .agg(" ".join, axis=1)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        raw["cluster_id"] = self._build_cluster_ids(gt, raw)

        return raw[["entity_id", "cluster_id", "record"]].copy()

    def _load_ground_truth_pairs(self, gt_path: Path) -> pd.DataFrame:
        if not gt_path.exists():
            raise FileNotFoundError(f"CDDB ground truth file not found: {gt_path}")

        gt = pd.read_csv(gt_path, dtype=str)
        gt = gt.fillna("")

        required = {"id1", "id2"}
        missing = required - set(gt.columns)
        if missing:
            raise ValueError(f"{gt_path.name} is missing columns: {sorted(missing)}")

        gt = gt[["id1", "id2"]].copy()
        gt["id1"] = gt["id1"].astype(str).str.strip()
        gt["id2"] = gt["id2"].astype(str).str.strip()
        gt = gt[(gt["id1"] != "") & (gt["id2"] != "")].reset_index(drop=True)
        if len(gt) == 0:
            raise ValueError(f"{gt_path.name} contains no valid id pairs")
        return gt

    def _build_cluster_ids(self, gt: pd.DataFrame, raw: pd.DataFrame) -> pd.Series:
        if "id1" not in gt.columns or "id2" not in gt.columns:
            raise ValueError("gt must contain columns: id1, id2")

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

        for left, right in gt[["id1", "id2"]].itertuples(index=False, name=None):
            left = str(left)
            right = str(right)
            union(left, right)

        root_to_cluster: dict[str, int] = {}
        cluster_ids: list[int] = []
        for entity_id in raw["entity_id"].astype(str):
            root = find(entity_id)
            if root not in root_to_cluster:
                root_to_cluster[root] = len(root_to_cluster)
            cluster_ids.append(root_to_cluster[root])

        return pd.Series(cluster_ids, index=raw.index, name="cluster_id")


class MusicBrainzAdapter(EntityTableAdapter):
    """Adapter for the MusicBrainz deduplication dataset (CSV)."""

    DEFAULT_FILENAME = "musicbrainz-20-A01.csv"
    DEFAULT_CORE_FIELDS = (
        "title",
        "artist",
        "album",
        "number",
        "length",
        "year",
        "language",
        "SourceID",
    )

    def __init__(self, raw_path: Path, filename: str | None = None):
        super().__init__(dataset_name="musicbrainz", raw_path=raw_path)
        self._filename = filename or self.DEFAULT_FILENAME

    def load_entity_table(self) -> pd.DataFrame:
        if self.raw_path.is_file():
            input_path = self.raw_path
        else:
            input_path = self.raw_path / self._filename

        if not input_path.exists():
            raise FileNotFoundError(
                f"MusicBrainz input file not found: {input_path}. "
                f"Expected a CSV file like '{self.DEFAULT_FILENAME}'."
            )

        raw = pd.read_csv(input_path, dtype=str)
        raw = raw.fillna("")

        required = {"TID", "CID"}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"{input_path.name} is missing columns: {sorted(missing)}")

        raw = raw.rename(columns={"TID": "entity_id", "CID": "cluster_id"})
        raw["entity_id"] = raw["entity_id"].astype(str).str.strip()

        if (raw["entity_id"] == "").any():
            raise ValueError(f"{input_path.name} contains empty TID values")

        text_columns = [c for c in self.DEFAULT_CORE_FIELDS if c in raw.columns]
        if not text_columns:
            # Fall back to all non-id fields.
            text_columns = [
                c
                for c in raw.columns
                if c not in {"entity_id", "cluster_id", "CTID", "id", "TID", "CID"}
            ]

        print(f"Using text fields for MusicBrainz: {text_columns}")

        raw["record"] = (
            raw[text_columns]
            .astype(str)
            .agg(" ".join, axis=1)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        raw["cluster_id"] = pd.to_numeric(raw["cluster_id"], errors="coerce")
        if raw["cluster_id"].isna().any():
            raise ValueError(
                f"{input_path.name} contains missing/non-numeric CID values"
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
    if dataset_name == "cddb":
        return CddbAdapter(raw_path)
    if dataset_name in {"musicbrainz", "music_brainz", "mb"}:
        return MusicBrainzAdapter(raw_path)
    raise ValueError(f"Unsupported dirty ER dataset: {dataset_name}")