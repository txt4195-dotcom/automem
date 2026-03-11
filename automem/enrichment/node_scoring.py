"""Node scoring via LLM — scores individual memories across 8 type-weight dimensions.

Each memory node gets continuous weights (0.0-1.0) for each knowledge type,
replacing the single-label type classification. A memory can be simultaneously
a pattern (0.3) and an insight (0.7) and a memory (0.9).

Scores are stored as node properties (w_pattern, w_insight, etc.) in FalkorDB
alongside Bayesian alpha/beta parameters for confidence tracking.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node weight dimensions — 8 knowledge types as continuous weights
# ---------------------------------------------------------------------------

NODE_WEIGHT_DIMENSIONS: Dict[str, str] = {
    "pattern": "Recurring behavior, approach, or regularity that repeats across situations",
    "insight": "Discovery, understanding, or learned knowledge — a-ha moment",
    "decision": "Strategic choice with rationale — why X over Y",
    "context": "Environmental, situational, or background information",
    "memory": "Factual record — what happened, what exists, raw data",
    "principle": "Fundamental rule, guideline, or axiom that governs behavior",
    "correction": "Fix, adjustment, or update to previous knowledge",
}

ALL_NODE_DIMENSIONS = list(NODE_WEIGHT_DIMENSIONS.keys())
TOTAL_NODE_DIMENSIONS = len(ALL_NODE_DIMENSIONS)  # 7


# ---------------------------------------------------------------------------
# Bayesian alpha/beta from score
# ---------------------------------------------------------------------------

def initial_alpha_beta(score: float) -> Tuple[float, float]:
    """Convert a nano score (0..1) into initial Bayesian alpha, beta.

    Same logic as edge_scoring: higher certainty (distance from 0.5)
    gets higher total evidence.
    """
    certainty = abs(score - 0.5) * 2  # 0..1
    total = 2 + certainty * 8         # 2..10
    alpha = total * score if score > 0 else 0.5
    beta = total * (1 - score) if score < 1 else 0.5
    return round(alpha, 2), round(beta, 2)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_node_scoring_prompt(memories: List[Dict[str, str]]) -> str:
    """Build the prompt for nano to score nodes across 8 type dimensions."""
    memory_block = "\n".join(
        f'  {m["id"][:8]}: "{m["content"][:400]}"'
        for m in memories
    )

    dimension_block = "\n".join(
        f"  {dim}: {desc}"
        for dim, desc in NODE_WEIGHT_DIMENSIONS.items()
    )

    return f"""Score each memory across 8 knowledge-type dimensions.
Each dimension is 0.0-1.0 where the value represents how strongly this memory embodies that type.
A memory can score high on multiple dimensions simultaneously.
0.0 = not at all this type, 1.0 = strongly this type.
Be precise — most memories are NOT every type. Score honestly.

Memories:
{memory_block}

Dimensions:
{dimension_block}

Return JSON only. Format:
{{
  "<memory_id_prefix>": {{
    "pattern": 0.1,
    "insight": 0.7,
    "decision": 0.0,
    "context": 0.2,
    "memory": 0.8,
    "principle": 0.3,
    "correction": 0.0
  }}
}}"""


# ---------------------------------------------------------------------------
# Data class for scored nodes
# ---------------------------------------------------------------------------

@dataclass
class NodeScores:
    """Scores for a single node across all 8 type dimensions."""
    memory_id: str
    weights: Dict[str, float] = field(default_factory=dict)
    alpha_beta: Dict[str, float] = field(default_factory=dict)

    def compute_alpha_beta(self) -> None:
        """Compute alpha/beta for all dimensions from weights."""
        self.alpha_beta = {}
        for dim, score in self.weights.items():
            a, b = initial_alpha_beta(score)
            self.alpha_beta[f"alpha_{dim}"] = a
            self.alpha_beta[f"beta_{dim}"] = b

    def node_properties(self) -> Dict[str, float]:
        """Return merged weight + alpha/beta properties for FalkorDB SET."""
        props: Dict[str, float] = {}
        for dim, score in self.weights.items():
            props[f"w_{dim}"] = score
        props.update(self.alpha_beta)
        return props


# ---------------------------------------------------------------------------
# Nano scorer
# ---------------------------------------------------------------------------

def score_nodes_with_llm(
    *,
    openai_client: Any,
    memories: List[Dict[str, Any]],
    model: str = "",
    batch_size: int = 10,
) -> List[NodeScores]:
    """Call LLM to score individual memory nodes across 8 type dimensions.

    Args:
        openai_client: Initialized OpenAI client
        memories: List of dicts with 'id' and 'content'
        model: LLM model to use (defaults to NODE_SCORING_MODEL or gpt-5-nano)
        batch_size: Max memories per LLM call

    Returns:
        List of NodeScores, one per memory that was successfully scored
    """
    if not model:
        model = os.getenv("NODE_SCORING_MODEL", "gpt-5-nano")

    if not memories:
        return []

    if openai_client is None:
        logger.warning("No OpenAI client available for node scoring")
        return []

    all_results: List[NodeScores] = []

    # Process in batches
    for i in range(0, len(memories), batch_size):
        batch = memories[i:i + batch_size]
        batch_results = _score_batch(openai_client, batch, model)
        all_results.extend(batch_results)

    return all_results


def _score_batch(
    openai_client: Any,
    memories: List[Dict[str, Any]],
    model: str,
) -> List[NodeScores]:
    """Score a single batch of memories."""
    prompt = _build_node_scoring_prompt(memories)

    # Map prefixes to full IDs
    prefix_to_full: Dict[str, str] = {}
    for m in memories:
        prefix_to_full[m["id"][:8]] = m["id"]

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a memory classifier. "
                        "Return ONLY valid JSON, no markdown, no explanation. "
                        "Score each dimension 0.0-1.0. "
                        "A memory can be multiple types simultaneously."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=4096,
            reasoning_effort="medium",
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        parsed = json.loads(raw)

    except json.JSONDecodeError:
        logger.exception("Failed to parse node scoring JSON response")
        return []
    except Exception:
        logger.exception("Node scoring LLM call failed")
        return []

    results: List[NodeScores] = []
    for prefix, dim_scores in parsed.items():
        full_id = prefix_to_full.get(prefix)
        if not full_id:
            for candidate_id in prefix_to_full.values():
                if candidate_id.startswith(prefix):
                    full_id = candidate_id
                    break
        if not full_id:
            logger.warning("Could not match prefix %s to any memory ID", prefix)
            continue

        # Validate and clamp scores
        weights: Dict[str, float] = {}
        for dim in ALL_NODE_DIMENSIONS:
            val = dim_scores.get(dim, 0.0)
            if isinstance(val, (int, float)):
                weights[dim] = max(0.0, min(1.0, float(val)))
            else:
                weights[dim] = 0.0

        ns = NodeScores(memory_id=full_id, weights=weights)
        ns.compute_alpha_beta()
        results.append(ns)

    logger.info(
        "Node scoring: %d/%d memories scored (model=%s)",
        len(results), len(memories), model,
    )
    return results


# ---------------------------------------------------------------------------
# FalkorDB persistence
# ---------------------------------------------------------------------------

def save_node_scores(graph: Any, scores: NodeScores) -> bool:
    """Write node scores to FalkorDB as Memory node properties.

    Sets w_pattern, w_insight, ..., alpha_pattern, beta_pattern, ... etc.
    """
    props = scores.node_properties()
    if not props:
        return False

    set_clauses = ", ".join(
        f"m.{key} = {val}" for key, val in props.items()
    )
    query = (
        f"MATCH (m:Memory {{id: $id}}) "
        f"SET {set_clauses}, m.node_scored = true, m.node_scored_at = $now"
    )

    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        graph.query(query, {"id": scores.memory_id, "now": now})
        return True
    except Exception:
        logger.exception("Failed to save node scores for %s", scores.memory_id)
        return False


def batch_score_and_save(
    *,
    openai_client: Any,
    graph: Any,
    memories: List[Dict[str, Any]],
    model: str = "",
) -> Dict[str, int]:
    """Score a list of memories and save results to FalkorDB.

    Returns dict with 'scored' and 'saved' counts.
    """
    scored = score_nodes_with_llm(
        openai_client=openai_client,
        memories=memories,
        model=model,
    )

    saved = 0
    for ns in scored:
        if save_node_scores(graph, ns):
            saved += 1

    return {"scored": len(scored), "saved": saved}
