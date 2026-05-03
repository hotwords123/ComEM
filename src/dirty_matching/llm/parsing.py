from __future__ import annotations

import json
import re
from typing import Any


def extract_json_payload(text: str) -> dict[str, Any]:
    stripped = str(text).strip()
    if not stripped:
        raise ValueError("Empty response content")

    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    candidate = fenced_match.group(1).strip() if fenced_match else stripped

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end < 0 or end <= start:
            raise
        payload = json.loads(candidate[start : end + 1])

    if not isinstance(payload, dict):
        raise ValueError("Response JSON must be an object")
    return payload


def safe_float(value: object, default: float = 0.5) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return max(0.0, min(1.0, out))


def normalize_cluster_lists(
    payload: dict[str, Any],
    expected_ids: list[str],
) -> list[list[str]]:
    raw_clusters = payload.get("clusters", [])
    if not isinstance(raw_clusters, list):
        raise ValueError("'clusters' must be a list")

    expected_set = set(expected_ids)
    seen: set[str] = set()
    clusters: list[list[str]] = []

    for item in raw_clusters:
        members_raw: list[object]
        if isinstance(item, list):
            members_raw = item
        elif isinstance(item, dict):
            members_raw = item.get("member_ids", [])
        else:
            continue

        if not isinstance(members_raw, list):
            continue

        cluster: list[str] = []
        for member in members_raw:
            member_id = str(member)
            if member_id not in expected_set:
                continue
            if member_id in seen:
                continue
            seen.add(member_id)
            cluster.append(member_id)
        if cluster:
            clusters.append(cluster)

    missing = [entity_id for entity_id in expected_ids if entity_id not in seen]
    for entity_id in missing:
        clusters.append([entity_id])

    if not clusters:
        return [[entity_id] for entity_id in expected_ids]
    return clusters
