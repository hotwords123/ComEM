from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

from diskcache import Cache
from jinja2 import Template

from src.dirty_matching.core.lookups import canonical_pair, truncate_text
from src.dirty_matching.llm.parsing import extract_json_payload, normalize_cluster_lists, safe_float
from src.utils import APICostCalculator, openai_chat_complete


class CommunityJointRefiner:
    system_prompt = (
        "You are a data cleaning expert for entity resolution. "
        "You must cluster records that refer to the same real-world entity. "
        "Return only strict JSON."
    )

    user_template = Template(
        """You are given one small community of potentially conflicting product records.
Cluster them into disjoint equivalence sets.

Return ONLY valid JSON with this schema:
{"clusters": [["id_1", "id_2"], ["id_3"]]}

Rules:
1) Use the exact entity ids.
2) Every id must appear exactly once.
3) No explanation text outside JSON.

Community ID: {{ community_id }}

Records:
{% for row in rows -%}
- [{{ row.id }}] {{ row.record }}
{% endfor %}
""" # + """
# Pairwise hints from baseline matcher (pred=True means baseline match):
# {% for hint in hints -%}
# - {{ hint }}
# {% endfor %}
# """
    )

    def __init__(
        self,
        model_name: str,
        max_retries: int,
        cache_dir: Path,
    ):
        self.model = model_name
        self.max_retries = max(1, int(max_retries))
        self.api_cost_decorator = APICostCalculator(model_name=model_name)
        cache = Cache(str(cache_dir))
        self.chat_complete = self.api_cost_decorator(
            cache.memoize(name="chat_complete")(openai_chat_complete)
        )

    def refine(
        self,
        community_id: int,
        entity_ids: list[str],
        record_lookup: dict[str, str],
        pair_lookup: dict[tuple[str, str], bool],
    ) -> dict[str, Any]:
        rows = [
            {
                "id": entity_id,
                "record": truncate_text(record_lookup.get(entity_id, ""), 320),
            }
            for entity_id in entity_ids
        ]

        hints: list[str] = []
        for left, right in combinations(entity_ids, 2):
            key = canonical_pair(left, right)
            pred = pair_lookup.get(key)
            if pred is None:
                relation = "unknown"
            else:
                relation = "pred=True" if bool(pred) else "pred=False"
            hints.append(f"{left} vs {right}: {relation}")
            if len(hints) >= 60:
                break

        prompt = self.user_template.render(
            community_id=community_id,
            rows=rows,
            hints=hints,
        )

        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.chat_complete(
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    model=self.model,
                    seed=42,
                    temperature=0.0,
                    max_tokens=5000,
                )
                content = str(response.choices[0].message.content or "")
                payload = extract_json_payload(content)
                clusters = normalize_cluster_lists(payload, entity_ids)
                return {
                    "community_id": int(community_id),
                    "clusters": clusters,
                    "status": "ok",
                    "attempts": int(attempt),
                }
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)

        return {
            "community_id": int(community_id),
            "clusters": [[entity_id] for entity_id in entity_ids],
            "status": "fallback_singletons",
            "attempts": int(self.max_retries),
            "error": last_error,
        }

    @property
    def cost(self) -> float:
        return float(self.api_cost_decorator.cost)


class GoldenRecordBuilder:
    system_prompt = (
        "You are an expert at building one canonical golden record from duplicate entities. "
        "Return only strict JSON."
    )

    user_template = Template(
        """Create one golden record for this local cluster.

Return ONLY valid JSON with this schema:
{"golden_record": "...", "confidence": 0.0}

Rules:
1) Merge complementary attributes and resolve contradictions conservatively.
2) Keep the output concise but complete.
3) confidence must be in [0,1].

Local cluster ID: {{ local_cluster_id }}

Records:
{% for row in rows -%}
- [{{ row.id }}] {{ row.record }}
{% endfor %}
"""
    )

    def __init__(
        self,
        model_name: str,
        max_retries: int,
        cache_dir: Path,
    ):
        self.model = model_name
        self.max_retries = max(1, int(max_retries))
        self.api_cost_decorator = APICostCalculator(model_name=model_name)
        cache = Cache(str(cache_dir))
        self.chat_complete = self.api_cost_decorator(
            cache.memoize(name="chat_complete")(openai_chat_complete)
        )

    def build(
        self,
        local_cluster_id: str,
        entity_ids: list[str],
        record_lookup: dict[str, str],
    ) -> dict[str, Any]:
        if len(entity_ids) == 1:
            only_id = entity_ids[0]
            return {
                "local_cluster_id": local_cluster_id,
                "golden_record": str(record_lookup.get(only_id, "")),
                "confidence": 1.0,
                "status": "singleton_passthrough",
                "attempts": 0,
            }

        rows = [
            {
                "id": entity_id,
                "record": truncate_text(record_lookup.get(entity_id, ""), 360),
            }
            for entity_id in entity_ids
        ]
        prompt = self.user_template.render(local_cluster_id=local_cluster_id, rows=rows)

        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.chat_complete(
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    model=self.model,
                    seed=42,
                    temperature=0.0,
                    max_tokens=800,
                )
                content = str(response.choices[0].message.content or "")
                payload = extract_json_payload(content)
                golden_record = str(payload.get("golden_record", "")).strip()
                if not golden_record:
                    raise ValueError("Missing 'golden_record' in response")
                return {
                    "local_cluster_id": local_cluster_id,
                    "golden_record": golden_record,
                    "confidence": safe_float(payload.get("confidence", 0.5), default=0.5),
                    "status": "ok",
                    "attempts": int(attempt),
                }
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)

        fallback = " | ".join(
            truncate_text(record_lookup.get(entity_id, ""), 160)
            for entity_id in entity_ids[:3]
        )
        return {
            "local_cluster_id": local_cluster_id,
            "golden_record": fallback,
            "confidence": 0.2,
            "status": "fallback_join",
            "attempts": int(self.max_retries),
            "error": last_error,
        }

    @property
    def cost(self) -> float:
        return float(self.api_cost_decorator.cost)
