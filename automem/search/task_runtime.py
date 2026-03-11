from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class TaskShard:
    index: int
    text: str


def split_turn_into_task_shards(turn_text: str) -> List[TaskShard]:
    """
    Split one user turn into multiple actionable task shards.

    This is intentionally small and deterministic for the first PoC:
    - preserve broad sentence boundaries
    - split on common "also/그리고" style connectors
    - trim filler fragments that should not become their own task
    """
    if not turn_text or not turn_text.strip():
        return []

    normalized = re.sub(r"\s+", " ", turn_text).strip()
    normalized = re.sub(
        r"\s*(그리고|그리고요|and also|also|plus)\s+",
        " || ",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"([?!。！？])\s+", r"\1 || ", normalized)

    raw_parts = [part.strip(" |") for part in normalized.split("||")]
    shards: List[TaskShard] = []
    for part in raw_parts:
        cleaned = _strip_filler_prefixes(part).strip()
        if not cleaned:
            continue
        if len(cleaned) < 2:
            continue
        shards.append(TaskShard(index=len(shards), text=cleaned))
    return shards


def rank_task_candidates(
    results: Sequence[Dict[str, Any]],
    *,
    persona_path: Sequence[str],
    trigger_family: str,
    desired_kind: str | None = None,
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for result in results:
        memory = result.get("memory") or {}
        metadata = memory.get("metadata") or {}
        semantic = _coerce_float(
            result.get("final_score", result.get("score", result.get("match_score", 0.0)))
        )
        boost = _metadata_weight(metadata, persona_path=persona_path, trigger_family=trigger_family)

        if desired_kind and metadata.get("memory_kind") == desired_kind:
            boost += 0.35

        scored = dict(result)
        scored["task_score"] = semantic + boost
        scored["task_score_components"] = {
            "semantic": semantic,
            "metadata_boost": boost,
        }
        ranked.append(scored)

    ranked.sort(
        key=lambda item: (
            -_coerce_float(item.get("task_score")),
            -_coerce_float(((item.get("memory") or {}).get("importance"))),
        )
    )
    return ranked


def select_weighted_source_list(
    results: Sequence[Dict[str, Any]],
    *,
    persona_path: Sequence[str],
    trigger_family: str,
) -> List[Dict[str, Any]]:
    ranked = rank_task_candidates(
        results,
        persona_path=persona_path,
        trigger_family=trigger_family,
        desired_kind="source_list",
    )
    if not ranked:
        return []

    best = ranked[0]
    metadata = ((best.get("memory") or {}).get("metadata") or {})
    source_rankings = metadata.get("source_rankings") or {}
    ranking = source_rankings.get(trigger_family) or source_rankings.get("default") or []
    entries = [dict(entry) for entry in ranking if isinstance(entry, dict)]
    if not entries:
        return []

    for entry in entries:
        source_name = str(entry.get("source") or "")
        entry["base_weight"] = _coerce_float(entry.get("weight"))
        entry["weight"] = entry["base_weight"] + _source_override_bonus(
            metadata,
            persona_path=persona_path,
            trigger_family=trigger_family,
            source_name=source_name,
        )

    entries.sort(key=lambda item: (-_coerce_float(item.get("weight")), str(item.get("source") or "")))
    return entries


def _strip_filler_prefixes(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(
        r"^(아\s*맞다\s*(그리고)?|oh right|right|also|and)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^아\s+", "", cleaned, flags=re.IGNORECASE)
    if re.fullmatch(r"아\s*맞다", cleaned, flags=re.IGNORECASE):
        return ""
    return cleaned


def _metadata_weight(
    metadata: Dict[str, Any],
    *,
    persona_path: Sequence[str],
    trigger_family: str,
) -> float:
    weight = _coerce_float(metadata.get("importance_base"))
    weight += _coerce_float((metadata.get("trigger_weights") or {}).get(trigger_family))

    persona_weights = metadata.get("persona_weights") or {}
    persona_trigger_weights = metadata.get("persona_trigger_weights") or {}

    for persona_name in _persona_candidates(persona_path):
        weight += _coerce_float(persona_weights.get(persona_name))
        weight += _coerce_float(persona_trigger_weights.get(f"{persona_name}:{trigger_family}"))

    return weight


def _source_override_bonus(
    metadata: Dict[str, Any],
    *,
    persona_path: Sequence[str],
    trigger_family: str,
    source_name: str,
) -> float:
    if not source_name:
        return 0.0

    overrides = metadata.get("persona_source_overrides") or {}
    bonus = 0.0
    source_key = source_name.strip().lower()

    for persona_name in _persona_candidates(persona_path):
        persona_bucket = overrides.get(persona_name) or {}
        trigger_bucket = persona_bucket.get(trigger_family) or {}
        bonus += _coerce_float(trigger_bucket.get(source_key))
    return bonus


def _persona_candidates(persona_path: Sequence[str]) -> Iterable[str]:
    seen: set[str] = set()
    for persona_name in reversed(list(persona_path)):
        cleaned = str(persona_name or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        yield cleaned


def _coerce_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
