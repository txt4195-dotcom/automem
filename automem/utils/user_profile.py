"""User profile utilities for personalized recall scoring.

User nodes store Beta distribution [a, b] per concept per category:
  lens_culture = [[8,2], [3,7], ...]   # [alpha, beta] pairs
  p_user = a / (a + b)                 # direction
  a + b = total evidence              # confidence

  [1, 1] = uniform prior (no opinion)
  Property absent = unobserved (skip in scoring)

Memory nodes store content stance as signed log-odds:
  lor_culture, lor_polity, etc.
  Each is [float, ...]. 0.0 = neutral, >0 = + pole, <0 = - pole.

Scoring: per dimension, compute p_agree (probability user and content agree).
  p_user = a / (a + b)
  p_node = sigmoid(content_lor)
  p_agree = p_user * p_node + (1-p_user) * (1-p_node)
  contribution = log(p_agree / (1-p_agree))  # align_lor
Sum across all dimensions.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from automem.utils.lens_concepts import (
    CATEGORY_KEYS,
    lens_property_name,
    lor_property_name,
)

logger = logging.getLogger(__name__)

DEFAULT_USER_ID = "user:default"
_MIN_LOR = 0.01  # skip near-zero content lors
_MIN_EVIDENCE = 2.01  # skip [1,1] uniform priors (a+b <= 2 = no evidence)
_P_EPS = 1e-6


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


# ── Graph access ─────────────────────────────────────────────

def get_user_lens(
    graph: Any,
    user_id: Optional[str] = None,
) -> Optional[Dict[str, List]]:
    """Fetch User node's lens ([a,b] arrays) from FalkorDB.

    Returns {"culture": [[8,2], [3,7], ...], "polity": [...], ...}
    or None if user not found. Falls back to default user.
    """
    uid = user_id or DEFAULT_USER_ID
    if graph is None:
        return None

    lens = _fetch_lens(graph, uid)
    if lens is not None:
        return lens

    if uid != DEFAULT_USER_ID:
        lens = _fetch_lens(graph, DEFAULT_USER_ID)
        if lens is not None:
            return lens

    return None


def _fetch_lens(graph: Any, uid: str) -> Optional[Dict[str, List]]:
    """Fetch all 6 lens properties for a user."""
    props = ", ".join(f"u.{lens_property_name(cat)}" for cat in CATEGORY_KEYS)
    query = f"MATCH (u:User {{id: $uid}}) RETURN {props}"

    try:
        result = graph.query(query, {"uid": uid})
        rows = getattr(result, "result_set", []) or []
        if not rows:
            return None

        row = rows[0]
        lens = {}
        has_any = False
        for i, cat in enumerate(CATEGORY_KEYS):
            val = row[i]
            if val is not None:
                lens[cat] = val
                has_any = True
        return lens if has_any else None

    except Exception:
        logger.exception("Failed to fetch user lens for %s", uid)
        return None


# ── Scoring ──────────────────────────────────────────────────

def compute_profile_score(
    lens: Dict[str, List],
    content_lor: Dict[str, List[float]],
) -> float:
    """Compute personalization score across all categories.

    lens values can be:
      - [[a,b], ...] (Beta pairs) — new format
      - [lor, ...] (flat lor) — legacy format, auto-detected

    Returns 0.0 when no signal.
    """
    if not lens or not content_lor:
        return 0.0

    total = 0.0
    for cat in CATEGORY_KEYS:
        cat_lens = lens.get(cat)
        cat_lor = content_lor.get(cat)
        if not cat_lens or not cat_lor:
            continue
        total += _score_category(cat_lens, cat_lor)

    return total


def _parse_ab(val: Any) -> Optional[Tuple[float, float]]:
    """Extract (a, b) from a lens value. Handles both [a,b] and scalar lor."""
    if isinstance(val, (list, tuple)) and len(val) == 2:
        try:
            a, b = float(val[0]), float(val[1])
            if a >= 0 and b >= 0:
                return (a, b)
        except (TypeError, ValueError):
            pass
    # Legacy: scalar lor — convert to approximate [a,b]
    if isinstance(val, (int, float)):
        p = _sigmoid(float(val))
        # Use n=10 as default evidence for legacy lor
        n = 10.0
        return (p * n, (1.0 - p) * n)
    return None


def _score_category(user_lens: List, content_lors: List[float]) -> float:
    """Score a single category using p_agree alignment."""
    n = min(len(user_lens), len(content_lors))
    score = 0.0

    for i in range(n):
        ab = _parse_ab(user_lens[i])
        if ab is None:
            continue

        a, b = ab
        # Skip if no real evidence (uniform prior)
        if a + b < _MIN_EVIDENCE:
            continue

        c_lor = content_lors[i]
        if not isinstance(c_lor, (int, float)) or abs(c_lor) < _MIN_LOR:
            continue

        # p_agree: probability user and content agree
        p_user = a / (a + b)
        p_node = _sigmoid(c_lor)
        p_agree = p_user * p_node + (1.0 - p_user) * (1.0 - p_node)

        # Clamp and convert to align_lor
        p_agree = max(_P_EPS, min(1.0 - _P_EPS, p_agree))
        align_lor = math.log(p_agree / (1.0 - p_agree))
        score += align_lor

    return score


# ── Memory lor extraction ────────────────────────────────────

def get_memory_resonance(memory: Dict[str, Any]) -> Optional[Dict[str, List[float]]]:
    """Extract lor dict from memory properties.

    Looks for lor_culture, lor_polity, etc.
    Returns {"culture": [...], ...} or None.
    """
    lor_dict = {}
    for cat in CATEGORY_KEYS:
        prop = lor_property_name(cat)
        val = memory.get(prop)
        if isinstance(val, list) and val:
            lor_dict[cat] = val

    return lor_dict if lor_dict else None
