"""Feedback-driven Beta lens update.

When a user gives feedback on a recall result (helpful / opposite_view),
update the user's [a,b] Beta priors on concept axes where the memory
has a strong stance.

Math per axis:
  lor -> p_node = sigmoid(lor)        # memory's stance direction
  stance = abs(p_node - 0.5) * 2      # how strong the stance is [0..1]
  novelty_damping = c / (c + a + b)   # light priors move more
  w = eta_base * stance * novelty_damping

  helpful (positive):
    a += w * p_node
    b += w * (1 - p_node)

  opposite_view (negative):
    a += w * (1 - p_node)
    b += w * p_node

  not_relevant: no update
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
from automem.utils.doctype_scoring import _sigmoid
from automem.utils.user_profile import _parse_ab

logger = logging.getLogger(__name__)

VALID_SIGNALS = {"helpful", "opposite_view", "not_relevant"}


def compute_lens_update(
    user_lens: Dict[str, List],
    memory_lors: Dict[str, List[float]],
    signal: str,
    *,
    eta_base: float = 1.0,
    top_k: int = 5,
    min_stance: float = 0.1,
    novelty_c: float = 2.0,
) -> Dict[str, Dict[str, Any]]:
    """Compute per-axis Beta updates from a feedback signal.

    Returns dict keyed by "category_index" (e.g. "culture_3"):
      {
        "cat": "culture",
        "idx": 3,
        "before": [a, b],
        "after": [a_new, b_new],
        "delta_p": p_new - p_old,
      }

    Returns empty dict for not_relevant or when no axes qualify.
    """
    if signal == "not_relevant" or signal not in VALID_SIGNALS:
        return {}

    if not user_lens or not memory_lors:
        return {}

    # Collect all candidate axes with their stance strength
    candidates: List[Tuple[str, int, float, float, float, float]] = []
    # (axis_key, idx, stance, p_node, a, b)

    for cat in CATEGORY_KEYS:
        cat_lens = user_lens.get(cat)
        cat_lor = memory_lors.get(cat)
        if not cat_lens or not cat_lor:
            continue

        n = min(len(cat_lens), len(cat_lor))
        for i in range(n):
            ab = _parse_ab(cat_lens[i])
            if ab is None:
                continue

            a, b = ab
            lor = cat_lor[i]
            if not isinstance(lor, (int, float)):
                continue

            p_node = _sigmoid(float(lor))
            stance = abs(p_node - 0.5) * 2.0

            if stance < min_stance:
                continue

            candidates.append((cat, i, stance, p_node, a, b))

    if not candidates:
        return {}

    # Top-K by stance strength
    candidates.sort(key=lambda c: c[2], reverse=True)
    selected = candidates[:top_k]

    is_positive = signal == "helpful"
    updates: Dict[str, Dict[str, Any]] = {}

    for cat, idx, stance, p_node, a, b in selected:
        novelty_damping = novelty_c / (novelty_c + a + b)
        w = eta_base * stance * novelty_damping

        if is_positive:
            a_new = a + w * p_node
            b_new = b + w * (1.0 - p_node)
        else:
            a_new = a + w * (1.0 - p_node)
            b_new = b + w * p_node

        p_old = a / (a + b) if (a + b) > 0 else 0.5
        p_new = a_new / (a_new + b_new) if (a_new + b_new) > 0 else 0.5

        axis_key = f"{cat}_{idx}"
        updates[axis_key] = {
            "cat": cat,
            "idx": idx,
            "before": [round(a, 4), round(b, 4)],
            "after": [round(a_new, 4), round(b_new, 4)],
            "delta_p": round(p_new - p_old, 6),
        }

    return updates


def apply_lens_update(
    graph: Any,
    user_id: str,
    updates: Dict[str, Dict[str, Any]],
) -> bool:
    """Write updated [a,b] values back to the User node in FalkorDB.

    Groups updates by category and issues one SET per category.
    Returns True if all writes succeeded.
    """
    if not updates:
        return True

    # Group by category
    by_cat: Dict[str, List[Tuple[int, float, float]]] = {}
    for axis_key, info in updates.items():
        cat = info["cat"]
        idx = info["idx"]
        a_new, b_new = info["after"]
        by_cat.setdefault(cat, []).append((idx, a_new, b_new))

    try:
        for cat, axis_updates in by_cat.items():
            prop = lens_property_name(cat)
            # Fetch current lens array for this category
            fetch_q = f"MATCH (u:User {{id: $uid}}) RETURN u.{prop}"
            result = graph.query(fetch_q, {"uid": user_id})
            rows = getattr(result, "result_set", []) or []
            if not rows or rows[0][0] is None:
                logger.warning(
                    "Cannot apply update: User %s has no %s property",
                    user_id, prop,
                )
                continue

            current = list(rows[0][0])  # copy

            for idx, a_new, b_new in axis_updates:
                if idx < len(current):
                    current[idx] = [round(a_new, 4), round(b_new, 4)]

            # Write back
            set_q = f"MATCH (u:User {{id: $uid}}) SET u.{prop} = $val"
            graph.query(set_q, {"uid": user_id, "val": current})

        return True

    except Exception:
        logger.exception("Failed to apply lens update for user %s", user_id)
        return False


def fetch_memory_lors(graph: Any, memory_id: str) -> Optional[Dict[str, List[float]]]:
    """Fetch lor_* properties from a Memory node."""
    props = ", ".join(f"m.{lor_property_name(cat)}" for cat in CATEGORY_KEYS)
    query = f"MATCH (m:Memory {{id: $mid}}) RETURN {props}"

    try:
        result = graph.query(query, {"mid": memory_id})
        rows = getattr(result, "result_set", []) or []
        if not rows:
            return None

        row = rows[0]
        lors: Dict[str, List[float]] = {}
        for i, cat in enumerate(CATEGORY_KEYS):
            val = row[i]
            if isinstance(val, list) and val:
                lors[cat] = val

        return lors if lors else None

    except Exception:
        logger.exception("Failed to fetch lors for memory %s", memory_id)
        return None
