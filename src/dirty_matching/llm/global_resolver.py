from __future__ import annotations

from pathlib import Path
from typing import Any

from diskcache import Cache
from jinja2 import Template

from src.dirty_matching.core.lookups import truncate_text
from src.dirty_matching.llm.parsing import extract_json_payload, normalize_cluster_lists, safe_float
from src.utils import APICostCalculator, openai_chat_complete


class GlobalConflictResolver:
    pair_system_prompt = (
        "You are an entity resolution judge. Decide whether two golden records refer to the same entity. "
        "Return only strict JSON."
    )
    pair_user_template = Template(
        """Return ONLY valid JSON with schema:
{"match": true, "confidence": 0.0}

Golden A ({{ a_id }}): {{ a_record }}
Golden B ({{ b_id }}): {{ b_record }}
"""
    )

    tri_system_prompt = (
        "You are an entity resolution judge. Resolve transitivity conflicts across three golden records. "
        "Return only strict JSON."
    )
    tri_user_template = Template(
        """Return ONLY valid JSON with schema:
{"clusters": [["{{ a_id }}", "{{ b_id }}"], ["{{ c_id }}"]]}

Golden records:
- {{ a_id }}: {{ a_record }}
- {{ b_id }}: {{ b_record }}
- {{ c_id }}: {{ c_record }}
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

    def resolve_pair(
        self,
        golden_a: str,
        golden_b: str,
        record_a: str,
        record_b: str,
    ) -> dict[str, Any]:
        prompt = self.pair_user_template.render(
            a_id=golden_a,
            b_id=golden_b,
            a_record=truncate_text(record_a, 420),
            b_record=truncate_text(record_b, 420),
        )

        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.chat_complete(
                    messages=[
                        {"role": "system", "content": self.pair_system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    model=self.model,
                    seed=42,
                    temperature=0.0,
                    max_tokens=220,
                )
                payload = extract_json_payload(str(response.choices[0].message.content or ""))
                return {
                    "pair": [golden_a, golden_b],
                    "match": bool(payload.get("match", False)),
                    "confidence": safe_float(payload.get("confidence", 0.5), default=0.5),
                    "status": "ok",
                    "attempts": int(attempt),
                }
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)

        return {
            "pair": [golden_a, golden_b],
            "match": False,
            "confidence": 0.0,
            "status": "fallback_non_match",
            "attempts": int(self.max_retries),
            "error": last_error,
        }

    def resolve_triangle(
        self,
        golden_ids: list[str],
        golden_lookup: dict[str, str],
    ) -> dict[str, Any]:
        if len(golden_ids) != 3:
            raise ValueError("Triangle resolver expects exactly 3 golden ids")

        a_id, b_id, c_id = golden_ids
        prompt = self.tri_user_template.render(
            a_id=a_id,
            b_id=b_id,
            c_id=c_id,
            a_record=truncate_text(golden_lookup.get(a_id, ""), 320),
            b_record=truncate_text(golden_lookup.get(b_id, ""), 320),
            c_record=truncate_text(golden_lookup.get(c_id, ""), 320),
        )

        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.chat_complete(
                    messages=[
                        {"role": "system", "content": self.tri_system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    model=self.model,
                    seed=42,
                    temperature=0.0,
                    max_tokens=300,
                )
                payload = extract_json_payload(str(response.choices[0].message.content or ""))
                clusters = normalize_cluster_lists(payload, golden_ids)
                return {
                    "golden_ids": golden_ids,
                    "clusters": clusters,
                    "status": "ok",
                    "attempts": int(attempt),
                }
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)

        return {
            "golden_ids": golden_ids,
            "clusters": [[gid] for gid in golden_ids],
            "status": "fallback_singletons",
            "attempts": int(self.max_retries),
            "error": last_error,
        }

    @property
    def cost(self) -> float:
        return float(self.api_cost_decorator.cost)
