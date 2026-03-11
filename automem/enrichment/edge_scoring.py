"""Edge scoring via LLM — scores relationships across 67 dimensions in 8 sub-layers.

Each dimension score is a float 0..1 representing how relevant that dimension is
to the relationship between two memories. Scores are stored as SCORE_* edge
properties in FalkorDB alongside Bayesian α/β parameters.

The scorer calls an LLM (default: gpt-4o-mini) once per batch of edges,
returning structured JSON with sub-layer scores for each edge.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dimension registry — 67 dimensions, 8 sub-layers
# ---------------------------------------------------------------------------

SCORE_EDGE_TYPES: Dict[str, List[str]] = {
    "SCORE_CULTURE": [
        "collectivism",       # 1. 집단주의 ↔ 개인주의 (Hofstede IDV)
        "hierarchy",          # 2. 위계 ↔ 수평 (Hofstede PDI)
        "uncertainty_avoid",  # 3. 불확실성 회피 ↔ 수용 (Hofstede UAI)
        "time_orientation",   # 4. 장기 ↔ 단기 (Hofstede LTO)
        "indulgence",         # 5. 향유 ↔ 절제 (Hofstede IVR)
        "context_comm",       # 6. 고맥락 ↔ 저맥락 (Hall)
    ],
    "SCORE_IDEOLOGY": [
        "liberalism",         # 7. 자유주의 ↔ 권위주의
        "progressivism",      # 8. 진보 ↔ 보수
        "market",             # 9. 자유시장 ↔ 규제/복지
        "growth_orient",      # 10. 성장 ↔ 지속가능
        "tribalism",          # 11. 부족주의/민족주의
        "dataism",            # 12. 인간중심 ↔ 데이터/알고리즘 (Harari)
        "feminism",           # 13. 가부장제 ↔ 성평등
        "environmentalism",   # 14. 인간중심 ↔ 생태중심
        "identity_politics",  # 15. 보편주의 ↔ 정체성/교차성
        "romanticism",        # 16. 감성·자연·개인감정
        "nihilism",           # 17. 의미 부정 ↔ 의미 구축
        "populism",           # 18. 엘리트 불신 ↔ 전문가 신뢰
        "anarchism",          # 19. 자율·탈권력 ↔ 제도·질서
        "transhumanism",      # 20. 인간 한계 수용 ↔ 기술적 초월
        "achievement",        # 21. 자기착취·성과주의 ↔ 여유 (Han)
        "stoicism",           # 22. 감정 절제 ↔ 감정 표현
        "consumerism",        # 23. 소비 = 정체성 ↔ 미니멀리즘
    ],
    "SCORE_RELIGION": [
        "religiosity",        # 24. 세속 ↔ 신앙
        "afterlife",          # 25. 현세 ↔ 내세
        "sacred_boundary",    # 26. 성역 유무
    ],
    "SCORE_BELIEF": [
        "empiricism",         # 27. 경험·데이터 ↔ 원칙·교리
        "humanism",           # 28. 인본 ↔ 기술/초월
        "rationalism",        # 29. 이성 ↔ 직관/감성
        "agency",             # 30. 자기결정 ↔ 운명/환경
        "existential",        # 31. 의미추구 ↔ 실용추구
        "moral_care",         # 32. care/fairness ↔ loyalty/authority (Haidt)
    ],
    "SCORE_SEX": [
        "same_attraction",    # 33. 동성 끌림 강도
        "other_attraction",   # 34. 이성 끌림 강도
        "gender_identity",    # 35. 시스젠더 ↔ 논바이너리/트랜스
        "sexuality_openness", # 36. 성적 보수 ↔ 성적 개방
    ],
    "SCORE_META": [
        "self_awareness",     # 37. 자기 인식 수준
        "bias_recognition",   # 38. 편향 인식 능력
        "feedback_quality",   # 39. 피드백 일관성
        "dunning_kruger",     # 40. 자기 능력 과대/과소 평가
        "confirmation_bias",  # 41. 확증편향 (Kahneman)
        "anchoring",          # 42. 앵커링 (Kahneman)
        "availability",       # 43. 가용성 편향 (Kahneman)
        "loss_aversion",      # 44. 손실회피 (Kahneman)
    ],
    "SCORE_COGNITIVE": [
        "openness",           # 45. 경험 개방성 (Big Five)
        "conscientiousness",  # 46. 성실성 (Big Five)
        "extraversion",       # 47. 외향성 (Big Five)
        "agreeableness",      # 48. 친화성 (Big Five)
        "neuroticism",        # 49. 신경성 (Big Five)
        "IQ",                 # 50. 논리·분석·추상
        "EQ",                 # 51. 감정 인식·공감
        "SQ",                 # 52. 사회적 맥락·관계 파악
        "narcissism",         # 53. 자기애 (Dark Triad)
        "machiavellianism",   # 54. 마키아벨리즘 (Dark Triad)
        "psychopathy",        # 55. 사이코패시 (Dark Triad)
        "expertise",          # 56. 초보 ↔ 전문가
    ],
    "SCORE_PROCESSING": [
        "temporal",           # 57. 최신 중시 ↔ 시간 무관
        "abstraction",        # 58. 구체적 ↔ 추상적
        "risk",               # 59. 안전 ↔ 모험
        "action",             # 60. 숙고 ↔ 실행
        "depth",              # 61. 깊이 ↔ 넓이
        "novelty",            # 62. 익숙한 것 ↔ 새로운 것
        "pragmatic",          # 63. 이론 ↔ 실용
        "autonomy",           # 64. 독립 ↔ 협업
        "curiosity",          # 65. 목적 지향 ↔ 탐색 지향
        "structure",          # 66. 자유 ↔ 체계
        "emotional_valence",  # 67. 긍정 편향 ↔ 위협 민감
    ],
}

# Overstory layers (gate the rest during recall)
OVERSTORY_LAYERS = [
    "SCORE_CULTURE",
    "SCORE_IDEOLOGY",
    "SCORE_RELIGION",
    "SCORE_BELIEF",
    "SCORE_SEX",
]

LOWER_LAYERS = [
    "SCORE_META",
    "SCORE_COGNITIVE",
    "SCORE_PROCESSING",
]

ALL_DIMENSIONS: List[str] = []
for _dims in SCORE_EDGE_TYPES.values():
    ALL_DIMENSIONS.extend(_dims)

TOTAL_DIMENSIONS = len(ALL_DIMENSIONS)  # 67


# ---------------------------------------------------------------------------
# Initial α/β from nano confidence
# ---------------------------------------------------------------------------

def initial_alpha_beta(score: float) -> Tuple[float, float]:
    """Convert a nano score (0..1) into initial Bayesian α, β.

    Higher score → higher α (more confident the score is correct).
    Total evidence (α+β) scales with distance from 0.5 — scores near 0.5
    get α=β=1 (maximum uncertainty), scores near 0 or 1 get higher total.
    """
    certainty = abs(score - 0.5) * 2  # 0..1
    total = 2 + certainty * 8         # 2..10
    alpha = total * score if score > 0 else 0.5
    beta = total * (1 - score) if score < 1 else 0.5
    return round(alpha, 2), round(beta, 2)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_scoring_prompt(
    source_content: str,
    neighbors: List[Dict[str, str]],
) -> str:
    """Build the prompt for nano to score edges across 8 sub-layers."""
    neighbor_block = "\n".join(
        f"  {n['id'][:8]}: \"{n['content'][:200]}\" (cosine: {n.get('cosine', '?')})"
        for n in neighbors
    )

    dimension_block = "\n".join(
        f"[{layer}] {', '.join(dims)}"
        for layer, dims in SCORE_EDGE_TYPES.items()
    )

    return f"""Score the relationship between a source memory and each neighbor across 8 sub-layers.
Each dimension is 0.0-1.0 where the value represents relevance/strength of that dimension to this specific relationship.
0.0 = completely irrelevant, 1.0 = highly relevant and defining.
Most scores will be low (0.0-0.2) — only score high when the dimension genuinely matters for understanding this relationship.

Source: "{source_content[:300]}"

Neighbors:
{neighbor_block}

Dimensions by sub-layer:
{dimension_block}

Return JSON only. Format:
{{
  "<neighbor_id_prefix>": {{
    "SCORE_CULTURE": {{"collectivism": 0.1, ...}},
    "SCORE_IDEOLOGY": {{"liberalism": 0.0, ...}},
    "SCORE_RELIGION": {{"religiosity": 0.0, ...}},
    "SCORE_BELIEF": {{"empiricism": 0.5, ...}},
    "SCORE_SEX": {{"same_attraction": 0.0, ...}},
    "SCORE_META": {{"self_awareness": 0.3, ...}},
    "SCORE_COGNITIVE": {{"openness": 0.4, ...}},
    "SCORE_PROCESSING": {{"pragmatic": 0.7, ...}}
  }}
}}"""


# ---------------------------------------------------------------------------
# Data class for scored edges
# ---------------------------------------------------------------------------

@dataclass
class EdgeScores:
    """Scores for a single edge across all 8 sub-layers."""
    source_id: str
    target_id: str
    scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # alpha/beta per dimension, keyed by f"alpha_{dim}" and f"beta_{dim}"
    alpha_beta: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def compute_alpha_beta(self) -> None:
        """Compute α/β for all dimensions from scores."""
        self.alpha_beta = {}
        for layer, dim_scores in self.scores.items():
            layer_ab: Dict[str, float] = {}
            for dim, score in dim_scores.items():
                a, b = initial_alpha_beta(score)
                layer_ab[f"alpha_{dim}"] = a
                layer_ab[f"beta_{dim}"] = b
            self.alpha_beta[layer] = layer_ab

    def edge_properties(self, layer: str) -> Dict[str, float]:
        """Return merged score + α/β properties for a FalkorDB edge."""
        props: Dict[str, float] = {}
        if layer in self.scores:
            props.update(self.scores[layer])
        if layer in self.alpha_beta:
            props.update(self.alpha_beta[layer])
        return props


# ---------------------------------------------------------------------------
# Nano scorer
# ---------------------------------------------------------------------------

def score_edges_with_llm(
    *,
    openai_client: Any,
    source_id: str,
    source_content: str,
    neighbors: List[Dict[str, Any]],
    model: str = "",
) -> List[EdgeScores]:
    """Call LLM to score edges between source and neighbors.

    Args:
        openai_client: Initialized OpenAI client
        source_id: ID of the source memory
        source_content: Content text of the source memory
        neighbors: List of dicts with 'id', 'content', and optionally 'cosine'
        model: LLM model to use (defaults to EDGE_SCORING_MODEL config)

    Returns:
        List of EdgeScores, one per neighbor that was successfully scored
    """
    import os
    if not model:
        model = os.getenv("EDGE_SCORING_MODEL", "gpt-5-nano")

    if not neighbors:
        return []

    if openai_client is None:
        logger.warning("No OpenAI client available for edge scoring")
        return []

    prompt = _build_scoring_prompt(source_content, neighbors)

    # Map neighbor ID prefixes back to full IDs
    prefix_to_full: Dict[str, str] = {}
    for n in neighbors:
        prefix_to_full[n["id"][:8]] = n["id"]

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a memory relationship scorer. "
                        "Return ONLY valid JSON, no markdown, no explanation. "
                        "Score each dimension 0.0-1.0. "
                        "Most dimensions should be 0.0 for any given relationship."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=4096,
            reasoning_effort="medium",
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        parsed = json.loads(raw)

    except json.JSONDecodeError:
        logger.exception("Failed to parse edge scoring JSON response")
        return []
    except Exception:
        logger.exception("Edge scoring LLM call failed")
        return []

    results: List[EdgeScores] = []
    for prefix, layer_scores in parsed.items():
        full_id = prefix_to_full.get(prefix)
        if not full_id:
            # Try matching by checking if any full ID starts with this prefix
            for candidate_id in prefix_to_full.values():
                if candidate_id.startswith(prefix):
                    full_id = candidate_id
                    break
        if not full_id:
            logger.debug("Edge scoring: unknown neighbor prefix %s", prefix)
            continue

        edge = EdgeScores(source_id=source_id, target_id=full_id)

        for layer, dims in SCORE_EDGE_TYPES.items():
            raw_layer = layer_scores.get(layer, {})
            if not isinstance(raw_layer, dict):
                continue
            # Only keep known dimensions, clamp to 0..1
            cleaned: Dict[str, float] = {}
            for dim in dims:
                val = raw_layer.get(dim)
                if val is not None:
                    try:
                        cleaned[dim] = max(0.0, min(1.0, float(val)))
                    except (TypeError, ValueError):
                        cleaned[dim] = 0.0
                else:
                    cleaned[dim] = 0.0
            edge.scores[layer] = cleaned

        edge.compute_alpha_beta()
        results.append(edge)

    logger.info(
        "Edge scoring: scored %d/%d neighbors for %s",
        len(results), len(neighbors), source_id[:8],
    )
    return results


# ---------------------------------------------------------------------------
# FalkorDB edge creation
# ---------------------------------------------------------------------------

def create_score_edges(
    *,
    graph: Any,
    edge_scores: List[EdgeScores],
    timestamp: str,
    logger_override: Any = None,
) -> int:
    """Create SCORE_* edges in FalkorDB for scored relationships.

    For each EdgeScores, creates up to 8 SCORE_* edges (one per sub-layer)
    between source and target nodes.

    Returns the number of edges created.
    """
    log = logger_override or logger
    created = 0

    for edge in edge_scores:
        for layer in SCORE_EDGE_TYPES:
            props = edge.edge_properties(layer)
            if not props:
                continue

            # Build SET clause dynamically from properties
            set_parts = [f"r.{k} = ${k}" for k in props]
            set_parts.append("r.created_at = $timestamp")

            params: Dict[str, Any] = {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "timestamp": timestamp,
            }
            params.update(props)

            query = f"""
                MATCH (a:Memory {{id: $source_id}})
                MATCH (b:Memory {{id: $target_id}})
                MERGE (a)-[r:{layer}]->(b)
                SET {', '.join(set_parts)}
            """

            try:
                graph.query(query, params)
                created += 1
            except Exception:
                log.exception(
                    "Failed to create %s edge %s->%s",
                    layer, edge.source_id[:8], edge.target_id[:8],
                )

    log.info("Created %d SCORE_* edges", created)
    return created
