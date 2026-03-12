"""POST /feedback — update User lens from recall feedback."""

from __future__ import annotations

import logging
from typing import Any, Callable

from flask import Blueprint, abort, jsonify, request

from automem.config import (
    FEEDBACK_ETA_BASE,
    FEEDBACK_MIN_STANCE,
    FEEDBACK_NOVELTY_C,
    FEEDBACK_TOP_K_LENSES,
)
from automem.utils.feedback_update import (
    VALID_SIGNALS,
    apply_lens_update,
    compute_lens_update,
    fetch_memory_lors,
)
from automem.utils.user_profile import DEFAULT_USER_ID, get_user_lens

logger = logging.getLogger(__name__)


def create_feedback_blueprint(
    get_memory_graph: Callable[[], Any],
) -> Blueprint:
    bp = Blueprint("feedback", __name__)

    @bp.route("/feedback", methods=["POST"])
    def post_feedback():
        data = request.get_json(silent=True) or {}

        memory_id = data.get("memory_id")
        signal = data.get("signal")
        user_id = data.get("user_id") or DEFAULT_USER_ID

        if not memory_id:
            abort(400, description="memory_id is required")
        if signal not in VALID_SIGNALS:
            abort(400, description=f"signal must be one of {sorted(VALID_SIGNALS)}")

        if signal == "not_relevant":
            return jsonify({
                "status": "skipped",
                "signal": signal,
                "reason": "not_relevant does not update lens",
            }), 200

        graph = get_memory_graph()
        if graph is None:
            abort(503, description="Graph database unavailable")

        # Fetch memory's lor values
        memory_lors = fetch_memory_lors(graph, memory_id)
        if not memory_lors:
            return jsonify({
                "status": "skipped",
                "signal": signal,
                "reason": "memory has no stance (lor) data",
            }), 200

        # Fetch user lens
        user_lens, resolved_user = get_user_lens(graph, user_id)
        if not user_lens:
            return jsonify({
                "status": "skipped",
                "signal": signal,
                "reason": f"no User lens found for {user_id}",
            }), 200

        # Compute updates
        updates = compute_lens_update(
            user_lens,
            memory_lors,
            signal,
            eta_base=FEEDBACK_ETA_BASE,
            top_k=FEEDBACK_TOP_K_LENSES,
            min_stance=FEEDBACK_MIN_STANCE,
            novelty_c=FEEDBACK_NOVELTY_C,
        )

        if not updates:
            return jsonify({
                "status": "skipped",
                "signal": signal,
                "reason": "no qualifying axes (all stances below threshold)",
            }), 200

        # Apply to graph
        success = apply_lens_update(graph, resolved_user, updates)

        summary = {
            "axes_updated": len(updates),
            "max_delta": max(abs(u["delta_p"]) for u in updates.values()),
        }

        logger.info(
            "feedback_applied",
            extra={
                "memory_id": memory_id,
                "signal": signal,
                "user_id": resolved_user,
                "axes_updated": summary["axes_updated"],
                "max_delta": summary["max_delta"],
            },
        )

        return jsonify({
            "status": "updated" if success else "partial",
            "signal": signal,
            "user_id": resolved_user,
            "updates": updates,
            "summary": summary,
        }), 200

    return bp
