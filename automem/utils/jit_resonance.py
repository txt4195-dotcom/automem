"""JIT (Just-In-Time) stance scoring at recall time.

When recall fetches candidate memories, some may not yet have stance
scores (lor_culture, lor_polity, etc.). This module identifies those
and scores them with nano before final ranking.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from automem.utils.lens_concepts import CATEGORY_KEYS, lor_property_name

logger = logging.getLogger(__name__)

# Max memories to JIT-score per recall (avoid latency spike)
JIT_BATCH_LIMIT = int(os.environ.get("JIT_RESONANCE_BATCH_LIMIT", "10"))


def needs_scoring(memory: Dict[str, Any]) -> bool:
    """Check if a memory is missing stance scores."""
    for cat in CATEGORY_KEYS:
        prop = lor_property_name(cat)
        if memory.get(prop) is not None:
            return False  # Has at least one category scored
    return True


# Backward compat alias
needs_resonance = needs_scoring


def jit_score_candidates(
    graph: Any,
    candidates: List[Dict[str, Any]],
) -> int:
    """Score stance for unscored candidates. Mutates candidates in-place.

    Returns count of newly scored memories.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return 0

    from automem.utils.resonance_scorer import score_and_save

    unscored = []
    for c in candidates:
        mem = c.get("memory", c)
        if needs_scoring(mem):
            unscored.append(c)

    if not unscored:
        return 0

    # Limit batch size to avoid latency
    batch = unscored[:JIT_BATCH_LIMIT]
    scored = 0

    for candidate in batch:
        mem = candidate.get("memory", candidate)
        memory_id = str(mem.get("id") or candidate.get("id") or "")
        content = str(mem.get("content") or "")

        if not memory_id or not content:
            continue

        lor_values = score_and_save(graph, memory_id, content)
        if lor_values is not None:
            # Update in-place so scoring picks it up
            for cat in CATEGORY_KEYS:
                prop = lor_property_name(cat)
                if cat in lor_values:
                    mem[prop] = lor_values[cat]
            scored += 1
            logger.debug("JIT scored stance for %s", memory_id)

    if scored:
        logger.info("JIT scored stance for %d/%d candidates", scored, len(unscored))

    return scored


def hydrate_lor_from_graph(
    graph: Any,
    candidates: List[Dict[str, Any]],
) -> int:
    """Fetch lor_* from FalkorDB and inject into candidate memory dicts.

    Recall results come from Qdrant payload which doesn't include lor_*.
    This batch-fetches lor values from FalkorDB so compute_profile_score
    can use them.

    Returns count of hydrated memories.
    """
    if graph is None:
        return 0

    # Collect IDs that are missing lor_*
    missing = []
    for c in candidates:
        mem = c.get("memory", c)
        if needs_scoring(mem):
            mid = str(mem.get("id") or c.get("id") or "")
            if mid:
                missing.append((mid, mem))

    if not missing:
        return 0

    # Batch fetch lor_* from FalkorDB
    lor_props = ", ".join(f"m.{lor_property_name(cat)}" for cat in CATEGORY_KEYS)
    ids = [mid for mid, _ in missing]

    try:
        result = graph.query(
            f"MATCH (m:Memory) WHERE m.id IN $ids RETURN m.id, {lor_props}",
            {"ids": ids},
        )
    except Exception:
        logger.debug("hydrate_lor_from_graph query failed", exc_info=True)
        return 0

    rows = getattr(result, "result_set", []) or []
    if not rows:
        return 0

    # Index by id
    lor_by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        mid = row[0]
        lor_data = {}
        for i, cat in enumerate(CATEGORY_KEYS):
            val = row[i + 1]
            if val is not None:
                # FalkorDB may return list or JSON string
                if isinstance(val, list):
                    lor_data[lor_property_name(cat)] = val
                elif isinstance(val, (bytes, str)):
                    import json as _json
                    try:
                        parsed = _json.loads(val)
                        if isinstance(parsed, list):
                            lor_data[lor_property_name(cat)] = parsed
                    except (ValueError, TypeError):
                        pass
        if lor_data:
            lor_by_id[mid] = lor_data

    # Inject into memory dicts
    hydrated = 0
    for mid, mem in missing:
        if mid in lor_by_id:
            mem.update(lor_by_id[mid])
            hydrated += 1

    if hydrated:
        logger.debug("Hydrated lor for %d/%d candidates from FalkorDB", hydrated, len(missing))

    return hydrated
