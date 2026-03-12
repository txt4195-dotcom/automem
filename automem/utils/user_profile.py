"""User profile utilities for personalized recall scoring.

v5: Named properties per concept.

User nodes store Beta distribution [a, b] per concept:
  lens_culture_individualism_collectivism = [8, 2]
  p_user = a / (a + b)                 # direction
  a + b = total evidence              # confidence
  [1, 1] = uniform prior (no opinion, skipped in scoring)
  Property absent = unobserved (skipped in scoring)

Memory nodes store content stance as signed log-odds:
  lor_culture_individualism_collectivism = 1.2
  0.0 = neutral, >0 = + pole, <0 = - pole.
  Property absent = unscored.

Scoring (log-space, no exp/sigmoid needed for ranking):
  alignment = (a - b) * lor
  Positive when user and content agree on direction.
  Sum across all dimensions → profile_score.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from automem.utils.lens_concepts import (
    ALL_CONCEPTS,
    ALL_LENS_PROPERTIES,
    ALL_LOR_PROPERTIES,
    LENS_PROPERTY_NAMES,
    LOR_PROPERTY_NAMES,
)

logger = logging.getLogger(__name__)

DEFAULT_USER_ID = "user:default"
_MIN_LOR = 0.01  # skip near-zero content lors
_MIN_EVIDENCE = 2.01  # skip [1,1] uniform priors (a+b <= 2 = no evidence)


# ── Graph access ─────────────────────────────────────────────

def get_user_lens(
    graph: Any,
    user_id: Optional[str] = None,
) -> Tuple[Optional[Dict[str, List]], Optional[str]]:
    """Fetch User node's lens (named [a,b] properties) from FalkorDB.

    Returns (lens_dict, resolved_user_id) tuple.
    lens_dict: {concept_name: [a, b], ...} or None.
    Only concepts with actual evidence (not [1,1]) are included.
    """
    uid = user_id or DEFAULT_USER_ID
    if graph is None:
        return None, None

    lens = _fetch_lens(graph, uid)
    if lens is not None:
        return lens, uid

    if uid != DEFAULT_USER_ID:
        lens = _fetch_lens(graph, DEFAULT_USER_ID)
        if lens is not None:
            return lens, DEFAULT_USER_ID

    return None, None


def _fetch_lens(graph: Any, uid: str) -> Optional[Dict[str, List]]:
    """Fetch all named lens properties for a user.

    Returns {concept_name: [a, b], ...} — only concepts that exist on the node.
    """
    props = ", ".join(f"u.{prop}" for prop in LENS_PROPERTY_NAMES)
    query = f"MATCH (u:User {{id: $uid}}) RETURN {props}"

    try:
        result = graph.query(query, {"uid": uid})
        rows = getattr(result, "result_set", []) or []
        if not rows:
            return None

        row = rows[0]
        lens = {}
        for i, concept in enumerate(ALL_CONCEPTS):
            val = row[i]
            if val is not None:
                lens[concept] = val
        return lens if lens else None

    except Exception:
        logger.exception("Failed to fetch user lens for %s", uid)
        return None


# ── Scoring ──────────────────────────────────────────────────

def compute_profile_score(
    lens: Dict[str, List],
    content_lor: Dict[str, float],
) -> float:
    """Compute personalization score across all concepts.

    lens: {concept_name: [a, b], ...}
    content_lor: {concept_name: lor_float, ...}

    Score per concept: (a - b) * lor
    Positive = user and content agree. Sum across all shared concepts.
    Skips uniform priors [1,1] and near-zero lors.
    """
    if not lens or not content_lor:
        return 0.0

    total = 0.0
    for concept, ab in lens.items():
        lor = content_lor.get(concept)
        if lor is None or abs(lor) < _MIN_LOR:
            continue

        parsed = _parse_ab(ab)
        if parsed is None:
            continue

        a, b = parsed
        if a + b < _MIN_EVIDENCE:
            continue

        total += (a - b) * lor

    return total


def _parse_ab(val: Any) -> Optional[Tuple[float, float]]:
    """Extract (a, b) from a lens value."""
    if isinstance(val, (list, tuple)) and len(val) == 2:
        try:
            a, b = float(val[0]), float(val[1])
            if a >= 0 and b >= 0:
                return (a, b)
        except (TypeError, ValueError):
            pass
    return None


# ── Memory lor extraction ────────────────────────────────────

def get_memory_resonance(memory: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Extract lor dict from memory properties (named property format).

    Looks for lor_culture_individualism_collectivism, etc.
    Returns {concept_name: lor_float, ...} or None.
    """
    lor_dict: Dict[str, float] = {}
    for concept, prop in ALL_LOR_PROPERTIES.items():
        val = memory.get(prop)
        if isinstance(val, (int, float)) and val != 0.0:
            lor_dict[concept] = float(val)

    return lor_dict if lor_dict else None
