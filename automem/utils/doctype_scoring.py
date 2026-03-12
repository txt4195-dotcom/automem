"""Doctype scoring: dynamic query-time intent matching against memory lor_doctype_*.

Unlike static User lens scoring (compute_profile_score), doctype scoring is
**dynamic** — the intent vector changes per query, not per user. It answers:
"does this memory match the kind of document the user wants right now?"

## Pipeline role

    recall candidates → [existing scores] → compute_doctype_score() → boosted ranking

This is a standalone pipeline stage (P5: name = contract). An orchestrator
composes it with profile scoring and other stages.

## Intent vector

    {"decision": 0.8, "pattern": 0.9, "insight": 0.3}

Keys are concept names from LENS_CATEGORIES["doctype"] (without _emphasis suffix
for convenience). Values are 0.0-1.0 weights representing how much the user
wants that document type right now.

## Scoring (v5: named properties)

    For each intent dimension:
        lor = m.lor_doctype_decision_emphasis  (named property)
        p_node = sigmoid(lor)   # how strongly this memory IS that type
        contribution = intent_weight * p_node
    score = sum of contributions

    Normalized by number of intent dimensions to keep scale comparable.

## Sources of intent

    1. Explicit: recall parameter `doctype_intent=decision:0.8,pattern:0.9`
    2. Auto-inferred: from query text (future — nano or embedding-based)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from automem.utils.lens_concepts import ALL_LOR_PROPERTIES, LENS_CATEGORIES

# Concept names in doctype category
DOCTYPE_CONCEPTS: List[str] = LENS_CATEGORIES["doctype"]

# Map short names (without _emphasis) to full concept name for convenience
_SHORT_TO_FULL: Dict[str, str] = {}
for _name in DOCTYPE_CONCEPTS:
    _SHORT_TO_FULL[_name] = _name
    if _name.endswith("_emphasis"):
        _SHORT_TO_FULL[_name[: -len("_emphasis")]] = _name


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def parse_doctype_intent(raw: str) -> Dict[str, float]:
    """Parse 'decision:0.8,pattern:0.9' into {concept_name: weight}.

    Accepts both full names (decision_emphasis) and short names (decision).
    Returns dict with full concept names as keys.
    """
    intent: Dict[str, float] = {}
    if not raw or not raw.strip():
        return intent

    for pair in raw.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        name, val_str = pair.split(":", 1)
        name = name.strip().lower()
        try:
            weight = float(val_str.strip())
            weight = max(0.0, min(1.0, weight))
        except (ValueError, TypeError):
            continue

        full_name = _SHORT_TO_FULL.get(name)
        if full_name is not None:
            intent[full_name] = weight

    return intent


def compute_doctype_score(
    query_intent: Dict[str, float],
    memory_lor: Dict[str, float],
) -> float:
    """Score how well a memory's doctype matches the query intent.

    Args:
        query_intent: {concept_name: weight} from parse_doctype_intent()
        memory_lor: {concept_name: lor_float, ...} (sparse, from resonance scorer)

    Returns:
        Score >= 0. Higher = better match. Returns 0.0 if no signal.
    """
    if not query_intent or not memory_lor:
        return 0.0

    score = 0.0
    n_dims = 0

    for concept_name, weight in query_intent.items():
        if weight <= 0.0:
            continue

        lor = memory_lor.get(concept_name)
        if lor is None or not isinstance(lor, (int, float)):
            continue

        p_node = _sigmoid(lor)
        score += weight * p_node
        n_dims += 1

    if n_dims > 0:
        score /= n_dims

    return score


def compute_doctype_score_from_memory(
    query_intent: Dict[str, float],
    memory: Dict[str, Any],
) -> float:
    """Convenience: extract named lor_doctype_* from memory dict and score.

    Memory dict has named properties like lor_doctype_decision_emphasis = 1.2.
    """
    lor: Dict[str, float] = {}
    for concept in DOCTYPE_CONCEPTS:
        prop = ALL_LOR_PROPERTIES[concept]
        val = memory.get(prop)
        if isinstance(val, (int, float)):
            lor[concept] = float(val)
        # Also check nested memory key (recall result format)
        if not lor.get(concept):
            inner = memory.get("memory", {})
            if isinstance(inner, dict):
                val = inner.get(prop)
                if isinstance(val, (int, float)):
                    lor[concept] = float(val)

    return compute_doctype_score(query_intent, lor)
