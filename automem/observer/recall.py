"""Observer-weighted recall — hierarchical re-ranking of recall results.

After the base recall scoring (vector + keyword + tag + importance etc.),
this module applies observer-dependent personalization by fetching SCORE_*
edges and computing hierarchical dot products with the observer vector.

The overstory gate makes this FASTER than naive scoring: if overstory
dimensions don't match, we skip fetching the remaining 3 sub-layers entirely.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from automem.enrichment.edge_scoring import (
    LOWER_LAYERS,
    OVERSTORY_LAYERS,
    SCORE_EDGE_TYPES,
)
from automem.observer.vector import (
    ObserverVector,
    compute_hierarchical_weight,
    get_or_create_observer,
)

logger = logging.getLogger(__name__)


def fetch_score_edges(
    *,
    graph: Any,
    source_id: str,
    target_id: str,
    layers: List[str],
) -> Dict[str, Dict[str, float]]:
    """Fetch SCORE_* edge properties between two memories.

    Args:
        graph: FalkorDB graph instance
        source_id: Source memory ID
        target_id: Target memory ID
        layers: List of SCORE_* edge type names to fetch

    Returns:
        Dict mapping layer name to dimension scores
    """
    result: Dict[str, Dict[str, float]] = {}

    for layer in layers:
        dims = SCORE_EDGE_TYPES.get(layer, [])
        if not dims:
            continue

        return_parts = ", ".join(f"r.{d} AS {d}" for d in dims)
        try:
            query_result = graph.query(
                f"""
                MATCH (a:Memory {{id: $src}})-[r:{layer}]->(b:Memory {{id: $tgt}})
                RETURN {return_parts}
                """,
                {"src": source_id, "tgt": target_id},
                timeout=2000,
            )

            if query_result.result_set:
                row = query_result.result_set[0]
                layer_scores: Dict[str, float] = {}
                for i, dim in enumerate(dims):
                    val = row[i] if i < len(row) else None
                    if val is not None:
                        try:
                            layer_scores[dim] = float(val)
                        except (TypeError, ValueError):
                            pass
                if layer_scores:
                    result[layer] = layer_scores

        except Exception:
            logger.debug("Failed to fetch %s edge %s->%s", layer, source_id[:8], target_id[:8])

    return result


def compute_observer_weight_for_edge(
    *,
    graph: Any,
    observer: ObserverVector,
    source_id: str,
    target_id: str,
    overstory_threshold: float = 0.1,
) -> Tuple[float, bool]:
    """Compute observer-dependent weight for a single edge.

    Uses hierarchical fetch: overstory first, gate check, then lower layers.

    Returns:
        (weight, was_gated) — weight 0..1, was_gated True if overstory blocked
    """
    # Phase 1: Fetch overstory SCORE_* edges only
    overstory_scores = fetch_score_edges(
        graph=graph,
        source_id=source_id,
        target_id=target_id,
        layers=OVERSTORY_LAYERS,
    )

    if not overstory_scores:
        # No SCORE_* edges exist yet (pre-v2 data) — return neutral weight
        return 0.5, False

    # Quick gate check on overstory
    overstory_total = 0.0
    overstory_count = 0
    for layer in OVERSTORY_LAYERS:
        layer_scores = overstory_scores.get(layer)
        if layer_scores:
            score = observer.dot_product(layer, layer_scores)
            overstory_total += score
            overstory_count += 1

    if overstory_count > 0:
        overstory_avg = overstory_total / overstory_count
        if overstory_avg < overstory_threshold:
            return 0.0, True  # Gated — don't even fetch lower layers

    # Phase 2: Fetch lower layers (only if overstory passed)
    lower_scores = fetch_score_edges(
        graph=graph,
        source_id=source_id,
        target_id=target_id,
        layers=LOWER_LAYERS,
    )

    all_scores = {**overstory_scores, **lower_scores}

    weight, _ = compute_hierarchical_weight(
        observer=observer,
        edge_scores_by_layer=all_scores,
        overstory_threshold=overstory_threshold,
    )

    return weight, False


def rerank_with_observer(
    *,
    graph: Any,
    results: List[Dict[str, Any]],
    observer_id: str,
    timestamp: str,
    observer_weight_factor: float = 0.3,
    overstory_threshold: float = 0.1,
) -> List[Dict[str, Any]]:
    """Re-rank recall results using observer-dependent scoring.

    Takes existing recall results (already scored by base scoring) and
    multiplies each score by an observer-dependent weight derived from
    the SCORE_* edges.

    The observer_weight_factor controls how much personalization affects
    the final score: 0.0 = no personalization, 1.0 = full personalization.

    Args:
        graph: FalkorDB graph instance
        results: List of recall result dicts (must have 'id' and 'score')
        observer_id: User/agent identifier for observer vector
        timestamp: UTC timestamp string
        observer_weight_factor: How much observer weighting affects final score
        overstory_threshold: Minimum overstory score before gating

    Returns:
        Re-ranked results list (same format, updated scores, sorted descending)
    """
    if not results or graph is None:
        return results

    observer = get_or_create_observer(
        graph=graph,
        observer_id=observer_id,
        timestamp=timestamp,
    )

    # For each result, find its SCORE_* edges and compute observer weight
    for result in results:
        memory_id = result.get("id", "")
        if not memory_id:
            continue

        # Check relations for SIMILAR_TO edges to find scored connections
        relations = result.get("relations", [])
        observer_weights: List[float] = []

        for rel in relations:
            rel_type = rel.get("type", "")
            if rel_type != "SIMILAR_TO":
                continue

            related_id = rel.get("memory", {}).get("id", "")
            if not related_id:
                continue

            weight, gated = compute_observer_weight_for_edge(
                graph=graph,
                observer=observer,
                source_id=memory_id,
                target_id=related_id,
                overstory_threshold=overstory_threshold,
            )

            if not gated:
                observer_weights.append(weight)

        # Compute average observer weight for this memory
        if observer_weights:
            avg_weight = sum(observer_weights) / len(observer_weights)
        else:
            avg_weight = 0.5  # Neutral if no scored edges

        # Blend base score with observer weight
        base_score = result.get("score", 0.0)
        personalized = base_score * (1 - observer_weight_factor) + \
                        base_score * avg_weight * observer_weight_factor

        result["score"] = personalized
        result["observer_weight"] = avg_weight
        result["observer_id"] = observer_id

    # Re-sort by updated score
    results.sort(key=lambda r: r.get("score", 0.0), reverse=True)

    return results
