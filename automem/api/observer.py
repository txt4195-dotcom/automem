"""Observer API — manage observer vectors and Bayesian feedback updates.

Endpoints:
  GET  /observer/<observer_id>         — Get current observer vector
  POST /observer/<observer_id>         — Create/update observer with priors
  POST /observer/<observer_id>/feedback — Submit feedback for Bayesian update
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from flask import Blueprint, abort, jsonify, request

logger = logging.getLogger(__name__)


def create_observer_blueprint(
    *,
    get_memory_graph_fn: Callable[[], Any],
    utc_now_fn: Callable[[], str],
    require_api_token_fn: Callable[[], None],
) -> Blueprint:
    """Create Flask blueprint for observer API endpoints."""

    bp = Blueprint("observer", __name__)

    @bp.route("/observer/<observer_id>", methods=["GET"])
    def get_observer(observer_id: str) -> Any:
        require_api_token_fn()
        graph = get_memory_graph_fn()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        from automem.observer.vector import load_observer

        observer = load_observer(graph=graph, observer_id=observer_id)
        if observer is None:
            abort(404, description=f"Observer '{observer_id}' not found")

        return jsonify({
            "status": "success",
            "observer_id": observer_id,
            "values": observer.values,
            "meta_cognition_score": observer.meta_cognition_score(),
        })

    @bp.route("/observer/<observer_id>", methods=["POST"])
    def create_or_update_observer(observer_id: str) -> Any:
        require_api_token_fn()
        graph = get_memory_graph_fn()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        data = request.get_json(silent=True) or {}
        nationality = data.get("nationality")
        role = data.get("role")

        from automem.observer.vector import get_or_create_observer, save_observer

        observer = get_or_create_observer(
            graph=graph,
            observer_id=observer_id,
            timestamp=utc_now_fn(),
            nationality=nationality,
            role=role,
        )

        return jsonify({
            "status": "success",
            "observer_id": observer_id,
            "values": observer.values,
            "meta_cognition_score": observer.meta_cognition_score(),
        })

    @bp.route("/observer/<observer_id>/feedback", methods=["POST"])
    def submit_feedback(observer_id: str) -> Any:
        """Submit recall feedback for Bayesian update.

        Request body:
        {
            "useful": ["memory_id_1", "memory_id_2"],
            "irrelevant": ["memory_id_3"],
            "query_memory_id": "source_memory_id"  // optional context
        }

        For each useful/irrelevant memory, updates both:
        1. Edge α/β (SCORE_* edges between query context and the memory)
        2. Observer vector (based on edge scores of useful/irrelevant memories)
        """
        require_api_token_fn()
        graph = get_memory_graph_fn()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        data = request.get_json(silent=True) or {}
        useful_ids: List[str] = data.get("useful", [])
        irrelevant_ids: List[str] = data.get("irrelevant", [])

        if not useful_ids and not irrelevant_ids:
            return jsonify({"status": "success", "updates": 0})

        from automem.enrichment.edge_scoring import SCORE_EDGE_TYPES
        from automem.observer.recall import fetch_score_edges
        from automem.observer.vector import (
            get_or_create_observer,
            save_observer,
        )

        timestamp = utc_now_fn()
        observer = get_or_create_observer(
            graph=graph,
            observer_id=observer_id,
            timestamp=timestamp,
        )

        meta_lr = observer.meta_cognition_score()
        all_layers = list(SCORE_EDGE_TYPES.keys())
        updates = 0

        def _process_feedback(memory_ids: List[str], positive: bool) -> int:
            count = 0
            for memory_id in memory_ids:
                # Fetch all SCORE_* edges connected to this memory
                try:
                    result = graph.query(
                        """
                        MATCH (m:Memory {id: $id})-[r:SIMILAR_TO]->(related:Memory)
                        RETURN related.id
                        LIMIT 5
                        """,
                        {"id": memory_id},
                    )
                except Exception:
                    continue

                for row in getattr(result, "result_set", []) or []:
                    if not row or not row[0]:
                        continue
                    related_id = str(row[0])

                    edge_scores = fetch_score_edges(
                        graph=graph,
                        source_id=memory_id,
                        target_id=related_id,
                        layers=all_layers,
                    )

                    if not edge_scores:
                        continue

                    # Update edge α/β
                    for layer, dim_scores in edge_scores.items():
                        for dim, score in dim_scores.items():
                            a_key = f"alpha_{dim}"
                            b_key = f"beta_{dim}"
                            increment = meta_lr * 1.0

                            if positive:
                                update_query = f"""
                                    MATCH (a:Memory {{id: $src}})-[r:{layer}]->(b:Memory {{id: $tgt}})
                                    SET r.{a_key} = coalesce(r.{a_key}, 1.0) + $inc
                                """
                            else:
                                update_query = f"""
                                    MATCH (a:Memory {{id: $src}})-[r:{layer}]->(b:Memory {{id: $tgt}})
                                    SET r.{b_key} = coalesce(r.{b_key}, 1.0) + $inc
                                """

                            try:
                                graph.query(
                                    update_query,
                                    {"src": memory_id, "tgt": related_id, "inc": increment},
                                )
                                count += 1
                            except Exception:
                                logger.debug(
                                    "Failed to update %s edge α/β for %s->%s",
                                    layer, memory_id[:8], related_id[:8],
                                )

                    # Update observer vector — useful=target 1.0, irrelevant=target 0.0
                    feedback_target = 1.0 if positive else 0.0
                    for layer, dim_scores in edge_scores.items():
                        for dim, score in dim_scores.items():
                            if score > 0.1:  # Only learn from non-trivial scores
                                observer.update_dimension(
                                    layer, dim,
                                    target=feedback_target,
                                    significance=score,
                                    lr=meta_lr,
                                )

            return count

        updates += _process_feedback(useful_ids, positive=True)
        updates += _process_feedback(irrelevant_ids, positive=False)

        # Save updated observer
        save_observer(graph=graph, observer=observer, timestamp=timestamp)

        return jsonify({
            "status": "success",
            "observer_id": observer_id,
            "updates": updates,
            "meta_cognition_score": observer.meta_cognition_score(),
        })

    return bp
