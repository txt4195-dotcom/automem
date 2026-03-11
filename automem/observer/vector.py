"""Observer vector — 67-dimensional continuous representation of a user.

Stored in FalkorDB as an Observer node with 8 sub-vectors (one per sub-layer),
each containing dimension values + Bayesian α/β parameters.

The observer vector is NOT directly asked to the user. It's inferred from:
1. Cold start priors (demographics → statistical averages)
2. Store behavior (what they choose to remember reveals values)
3. Recall feedback (what they find useful/irrelevant)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from automem.enrichment.edge_scoring import (
    ALL_DIMENSIONS,
    LOWER_LAYERS,
    OVERSTORY_LAYERS,
    SCORE_EDGE_TYPES,
)

logger = logging.getLogger(__name__)

# Default cold start — neutral on everything (0.5 = no opinion)
DEFAULT_SCORE = 0.5
DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 1.0


# ---------------------------------------------------------------------------
# Cold start priors — demographic → statistical averages
# ---------------------------------------------------------------------------

# Hofstede cultural dimension scores for common nationalities
# Source: https://www.hofstede-insights.com/country-comparison/
# Normalized to 0..1 scale (original 0..100 → 0..1)
HOFSTEDE_PRIORS: Dict[str, Dict[str, float]] = {
    "korean": {
        "collectivism": 0.82,       # IDV 18 → high collectivism
        "hierarchy": 0.60,          # PDI 60
        "uncertainty_avoid": 0.85,  # UAI 85
        "time_orientation": 1.00,   # LTO 100
        "indulgence": 0.29,         # IVR 29
        "context_comm": 0.85,       # Hall: high-context
    },
    "japanese": {
        "collectivism": 0.54,
        "hierarchy": 0.54,
        "uncertainty_avoid": 0.92,
        "time_orientation": 0.88,
        "indulgence": 0.42,
        "context_comm": 0.90,
    },
    "american": {
        "collectivism": 0.09,       # IDV 91 → very individualistic
        "hierarchy": 0.40,
        "uncertainty_avoid": 0.46,
        "time_orientation": 0.26,
        "indulgence": 0.68,
        "context_comm": 0.20,       # Hall: low-context
    },
}

# Role-based priors for processing/cognitive dimensions
ROLE_PRIORS: Dict[str, Dict[str, float]] = {
    "developer": {
        "IQ": 0.75,
        "pragmatic": 0.85,
        "depth": 0.65,
        "structure": 0.70,
        "empiricism": 0.80,
        "rationalism": 0.75,
        "openness": 0.70,
        "curiosity": 0.75,
    },
    "designer": {
        "openness": 0.85,
        "abstraction": 0.70,
        "EQ": 0.70,
        "novelty": 0.75,
        "pragmatic": 0.60,
    },
    "researcher": {
        "IQ": 0.80,
        "depth": 0.85,
        "empiricism": 0.90,
        "rationalism": 0.80,
        "openness": 0.80,
        "curiosity": 0.90,
    },
}


@dataclass
class ObserverVector:
    """67-dimensional observer vector with per-dimension α/β."""

    observer_id: str  # user/agent identifier
    # Sub-layer → dimension → value
    values: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Sub-layer → dimension → (alpha, beta)
    alpha_beta: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=dict)

    @classmethod
    def create_default(cls, observer_id: str) -> "ObserverVector":
        """Create a default observer with neutral values and low confidence."""
        vec = cls(observer_id=observer_id)
        for layer, dims in SCORE_EDGE_TYPES.items():
            vec.values[layer] = {dim: DEFAULT_SCORE for dim in dims}
            vec.alpha_beta[layer] = {
                dim: (DEFAULT_ALPHA, DEFAULT_BETA) for dim in dims
            }
        return vec

    @classmethod
    def create_with_priors(
        cls,
        observer_id: str,
        *,
        nationality: Optional[str] = None,
        role: Optional[str] = None,
    ) -> "ObserverVector":
        """Create observer with demographic cold-start priors."""
        vec = cls.create_default(observer_id)

        # Apply Hofstede cultural priors
        if nationality and nationality.lower() in HOFSTEDE_PRIORS:
            priors = HOFSTEDE_PRIORS[nationality.lower()]
            for dim, val in priors.items():
                if dim in vec.values.get("SCORE_CULTURE", {}):
                    vec.values["SCORE_CULTURE"][dim] = val
                    # Hofstede data is well-established → moderate confidence
                    vec.alpha_beta["SCORE_CULTURE"][dim] = (3.0, 3.0)

        # Apply role-based priors
        if role and role.lower() in ROLE_PRIORS:
            priors = ROLE_PRIORS[role.lower()]
            for dim, val in priors.items():
                # Find which layer this dimension belongs to
                for layer, dims in SCORE_EDGE_TYPES.items():
                    if dim in dims:
                        vec.values[layer][dim] = val
                        # Role priors are weaker than cultural → low confidence
                        vec.alpha_beta[layer][dim] = (2.0, 2.0)
                        break

        return vec

    def get_layer_vector(self, layer: str) -> Dict[str, float]:
        """Get the value vector for a specific sub-layer."""
        return self.values.get(layer, {})

    def dot_product(self, layer: str, edge_scores: Dict[str, float]) -> float:
        """Compute dot product between observer sub-vector and edge scores."""
        obs_vals = self.values.get(layer, {})
        if not obs_vals or not edge_scores:
            return 0.0

        total = 0.0
        count = 0
        for dim, obs_val in obs_vals.items():
            edge_val = edge_scores.get(dim)
            if edge_val is not None:
                total += obs_val * edge_val
                count += 1

        return total / count if count > 0 else 0.0

    def meta_cognition_score(self) -> float:
        """Average meta-cognition score — used as feedback learning rate."""
        meta = self.values.get("SCORE_META", {})
        if not meta:
            return 0.5
        return sum(meta.values()) / len(meta)

    def update_dimension(
        self,
        layer: str,
        dimension: str,
        target: float,
        significance: float = 1.0,
        lr: float = 1.0,
    ) -> float:
        """Bayesian pseudo-observation update toward a continuous target.

        Adds fractional pseudo-observations proportional to (lr * significance):
            α += weight * target
            β += weight * (1 - target)
        Value converges to target; α+β grows as confidence accumulates.

        Args:
            layer: Sub-layer (e.g., "SCORE_IDEOLOGY")
            dimension: Dimension name (e.g., "liberalism")
            target: Target value in [0, 1] (e.g., nano edge score)
            significance: Error magnitude — how much to learn (like importance)
            lr: Base learning rate multiplier

        Returns:
            Delta applied to the dimension value (new - old). 0.0 if skipped.
        """
        if layer not in self.alpha_beta or dimension not in self.alpha_beta[layer]:
            return 0.0

        old_value = self.values.get(layer, {}).get(dimension, DEFAULT_SCORE)

        alpha, beta = self.alpha_beta[layer][dimension]
        weight = lr * significance

        alpha += weight * target
        beta += weight * (1.0 - target)

        self.alpha_beta[layer][dimension] = (round(alpha, 3), round(beta, 3))
        new_value = round(alpha / (alpha + beta), 4)
        self.values[layer][dimension] = new_value

        return round(new_value - old_value, 6)


# ---------------------------------------------------------------------------
# FalkorDB persistence
# ---------------------------------------------------------------------------

def save_observer(*, graph: Any, observer: ObserverVector, timestamp: str) -> bool:
    """Save or update observer vector in FalkorDB as an Observer node."""
    try:
        # Flatten all values and α/β into node properties
        props: Dict[str, Any] = {
            "observer_id": observer.observer_id,
            "updated_at": timestamp,
        }

        for layer, dim_vals in observer.values.items():
            for dim, val in dim_vals.items():
                props[f"{layer}_{dim}"] = val

        for layer, dim_ab in observer.alpha_beta.items():
            for dim, (a, b) in dim_ab.items():
                props[f"{layer}_alpha_{dim}"] = a
                props[f"{layer}_beta_{dim}"] = b

        # Build dynamic SET clause
        set_parts = [f"o.{k} = ${k}" for k in props if k != "observer_id"]

        graph.query(
            f"""
            MERGE (o:Observer {{observer_id: $observer_id}})
            SET {', '.join(set_parts)}
            """,
            props,
        )
        return True

    except Exception:
        logger.exception("Failed to save observer %s", observer.observer_id)
        return False


def load_observer(*, graph: Any, observer_id: str) -> Optional[ObserverVector]:
    """Load observer vector from FalkorDB."""
    try:
        result = graph.query(
            "MATCH (o:Observer {observer_id: $id}) RETURN o",
            {"id": observer_id},
        )
        if not result.result_set:
            return None

        node = result.result_set[0][0]
        props = getattr(node, "properties", {})
        if not isinstance(props, dict):
            props = dict(getattr(node, "__dict__", {}))

        vec = ObserverVector(observer_id=observer_id)

        for layer, dims in SCORE_EDGE_TYPES.items():
            layer_vals: Dict[str, float] = {}
            layer_ab: Dict[str, Tuple[float, float]] = {}

            for dim in dims:
                val_key = f"{layer}_{dim}"
                a_key = f"{layer}_alpha_{dim}"
                b_key = f"{layer}_beta_{dim}"

                val = props.get(val_key)
                layer_vals[dim] = float(val) if val is not None else DEFAULT_SCORE

                a = props.get(a_key)
                b = props.get(b_key)
                layer_ab[dim] = (
                    float(a) if a is not None else DEFAULT_ALPHA,
                    float(b) if b is not None else DEFAULT_BETA,
                )

            vec.values[layer] = layer_vals
            vec.alpha_beta[layer] = layer_ab

        return vec

    except Exception:
        logger.exception("Failed to load observer %s", observer_id)
        return None


def get_or_create_observer(
    *,
    graph: Any,
    observer_id: str,
    timestamp: str,
    nationality: Optional[str] = None,
    role: Optional[str] = None,
) -> ObserverVector:
    """Load existing observer or create with cold-start priors."""
    existing = load_observer(graph=graph, observer_id=observer_id)
    if existing is not None:
        return existing

    if nationality or role:
        vec = ObserverVector.create_with_priors(
            observer_id, nationality=nationality, role=role,
        )
    else:
        vec = ObserverVector.create_default(observer_id)

    save_observer(graph=graph, observer=vec, timestamp=timestamp)
    logger.info(
        "Created new observer %s (nationality=%s, role=%s)",
        observer_id, nationality, role,
    )
    return vec


# ---------------------------------------------------------------------------
# Hierarchical recall scoring
# ---------------------------------------------------------------------------

def compute_hierarchical_weight(
    *,
    observer: ObserverVector,
    edge_scores_by_layer: Dict[str, Dict[str, float]],
    overstory_threshold: float = 0.1,
) -> Tuple[float, Dict[str, float]]:
    """Compute observer-weighted edge importance using hierarchical multiplication.

    Overstory layers (culture, ideology, religion, belief, sex) act as gates:
    if the combined overstory score is below threshold, weight is 0 regardless
    of lower layer scores.

    Returns:
        (final_weight, component_scores) where component_scores maps
        layer name to its dot-product contribution.
    """
    components: Dict[str, float] = {}

    # Phase 1: Overstory gate
    overstory_total = 0.0
    overstory_count = 0
    for layer in OVERSTORY_LAYERS:
        layer_scores = edge_scores_by_layer.get(layer, {})
        if layer_scores:
            score = observer.dot_product(layer, layer_scores)
            components[layer] = score
            overstory_total += score
            overstory_count += 1

    overstory_avg = overstory_total / overstory_count if overstory_count > 0 else 0.5
    components["overstory_avg"] = overstory_avg

    if overstory_avg < overstory_threshold:
        components["gated"] = True
        return 0.0, components

    # Phase 2: Lower layers
    lower_total = 0.0
    lower_count = 0
    for layer in LOWER_LAYERS:
        layer_scores = edge_scores_by_layer.get(layer, {})
        if layer_scores:
            score = observer.dot_product(layer, layer_scores)
            components[layer] = score
            lower_total += score
            lower_count += 1

    lower_avg = lower_total / lower_count if lower_count > 0 else 0.5
    components["lower_avg"] = lower_avg

    # Hierarchical multiplication: overstory gates, lower layers modulate
    final = overstory_avg * lower_avg
    components["gated"] = False

    return final, components


# ---------------------------------------------------------------------------
# Store reverse inference
# ---------------------------------------------------------------------------

def infer_observer_from_store(
    *,
    observer: ObserverVector,
    edge_scores_list: List[Dict[str, Dict[str, float]]],
    significance: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Backpropagate observer values from what the user chose to store.

    nano edge scores = target (immutable base, ground truth estimate).
    observer values = current estimate (moves toward target).
    significance = error magnitude (memory importance → gradient scale).

    When the user stores with high importance but the observer profile says
    it shouldn't matter — that mismatch IS the error signal.

    Returns:
        Delta dict: {layer: {dim: delta}} for transparency/debugging.
    """
    meta_lr = observer.meta_cognition_score()
    deltas: Dict[str, Dict[str, float]] = {}

    for edge_scores in edge_scores_list:
        for layer, dim_scores in edge_scores.items():
            if layer not in SCORE_EDGE_TYPES:
                continue
            layer_deltas = deltas.setdefault(layer, {})
            for dim, target in dim_scores.items():
                if target < 0.1:  # Skip near-zero targets (no signal)
                    continue
                delta = observer.update_dimension(
                    layer, dim,
                    target=target,
                    significance=significance,
                    lr=meta_lr,
                )
                if abs(delta) > 1e-6:
                    layer_deltas[dim] = round(delta, 6)

    return deltas
