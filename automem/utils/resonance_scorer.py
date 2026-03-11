"""Stance scoring: 48 concept dimensions via nano LLM.

Each memory gets scored on 48 bipolar concept dimensions across 6 categories
(culture, polity, economy, epistemic, moral, religion). Nano analyzes the
text per category, producing evidence-grounded stance judgments.

## Nano output format

    {"concept_key": {"e": "quote from text", "d": "pole label", "s": 0.9, "c": 0.9}}

    - e: exact phrase from the text (evidence)
    - d: which pole label the evidence supports (copied from input)
    - s: strength 0.0-1.0 (how strongly the text leans that way)
    - c: confidence 0.0-1.0 (how sure the model is)
    - omit: no signal for this dimension

## Pole matching (d → sign)

    Nano writes a pole label in "d". We match it to plus/minus to determine sign:
    1. Exact match against full label
    2. Substring containment (d in label, unambiguous)
    3. Word overlap (excluding shared words between plus/minus)
    4. Embedding cosine similarity fallback (cached pole embeddings)
    5. Discard if all methods fail (ambiguous)

## Conversion to lor

    signed_s = s if plus, -s if minus
    lor = logit((s + 1) / 2)   # maps -1..+1 → -inf..+inf

## Storage

    FalkorDB Memory node properties: lor_culture, lor_polity, etc.
    Each is a float array matching LENS_CATEGORIES[cat] order.
    0.0 = neutral / not scored.

## Pipeline integration

    - Enrichment worker: scores on store, retries on failure
    - Backfill: re-enrichment scores already-processed memories missing lor_*
    - JIT: recall-time fallback for unscored candidates (jit_resonance.py)

## User alignment (user_profile.py)

    User lens is [[a,b], ...] Beta pairs per concept.
    p_agree = p_user * p_node + (1-p_user) * (1-p_node)
    align_lor = logit(p_agree)
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np

from automem.utils.lens_concepts import (
    ALL_CONCEPTS,
    CATEGORY_KEYS,
    CONCEPT_LOCATION,
    CONCEPT_POLES,
    LENS_CATEGORIES,
    lor_property_name,
)

logger = logging.getLogger(__name__)

# ── Pole label embedding cache ──────────────────────────────
# Pre-computed embeddings for pole labels, built lazily on first use.

_pole_embed_cache: Dict[str, Any] = {}  # label -> numpy array
_EMBED_SIM_THRESHOLD = 0.3  # below this, both poles are too far → discard


def _get_embedding_provider():
    """Get the embedding provider (lazy import to avoid circular deps)."""
    try:
        from automem.embedding import get_provider
        return get_provider()
    except Exception:
        return None


def _embed_text(text: str) -> Optional[Any]:
    """Embed a short text, using cache for pole labels."""
    if text in _pole_embed_cache:
        return _pole_embed_cache[text]
    provider = _get_embedding_provider()
    if provider is None:
        return None
    try:
        vec = provider.embed(text)
        arr = np.array(vec, dtype=np.float32)
        _pole_embed_cache[text] = arr
        return arr
    except Exception:
        return None


def _cosine_sim(a, b) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


def _match_pole_by_embedding(
    d_text: str, plus_label: str, minus_label: str,
) -> Optional[bool]:
    """Use embedding cosine similarity to decide if d matches plus or minus.

    Returns True if minus, False if plus, None if ambiguous (both low sim).
    """
    d_vec = _embed_text(d_text)
    if d_vec is None:
        return None
    plus_vec = _embed_text(plus_label)
    minus_vec = _embed_text(minus_label)
    if plus_vec is None or minus_vec is None:
        return None

    plus_sim = _cosine_sim(d_vec, plus_vec)
    minus_sim = _cosine_sim(d_vec, minus_vec)

    # Both too low → can't tell
    if max(plus_sim, minus_sim) < _EMBED_SIM_THRESHOLD:
        logger.debug(
            "Embedding match too low: d=%r plus_sim=%.3f minus_sim=%.3f",
            d_text, plus_sim, minus_sim,
        )
        return None

    return minus_sim > plus_sim

# ── Math helpers ─────────────────────────────────────────────

_P_EPS = 1e-6  # clamp p away from 0/1 before logit


def p_to_lor(p: float) -> float:
    """Convert p_plus to signed log-odds (logit).

    lor = log(p / (1-p)). Positive = leans + pole.
    """
    p = max(_P_EPS, min(1.0 - _P_EPS, p))
    return math.log(p / (1.0 - p))


def lor_to_p(lor: float) -> float:
    """Convert signed log-odds to probability (sigmoid)."""
    return 1.0 / (1.0 + math.exp(-lor))


# ── Nano prompt ──────────────────────────────────────────────

STANCE_SYSTEM_PROMPT = """You receive a text and a JSON of conceptual dimensions (each with "plus" and "minus" poles).

For each dimension the text addresses, output:
{"key": {"e": "quote from text", "d": "pole label", "s": 0.0-1.0, "c": 0.0-1.0}}

- e: exact phrase from the text (evidence)
- d: copy the "plus" or "minus" label that the evidence supports
- s: strength (0.3 = mild, 0.6 = clear, 0.9 = central topic)
- c: confidence (0.3 = guess, 0.7 = likely, 0.9 = certain)

Example:
Input dimensions: {"certainty_ambiguity": {"plus": "certainty", "minus": "ambiguity"}}
Text: "reject certainty, embrace the unknown"
Output: {"certainty_ambiguity": {"e": "reject certainty, embrace the unknown", "d": "ambiguity", "s": 0.9, "c": 0.9}}

ONLY include dimensions with clear textual evidence. Omit the rest. JSON only."""


CATEGORY_CONTEXT: Dict[str, str] = {
    "culture": "Cultural values: how people relate to each other and society",
    "polity": "Political orientation: governance, institutions, social order",
    "economy": "Economic values: markets, work, consumption, environment",
    "epistemic": "Ways of knowing: reason, evidence, certainty, technology",
    "moral": "Moral foundations: what matters ethically (Haidt framework)",
    "religion": "Religious and spiritual orientation",
}


def _build_user_prompt(content: str, concepts: List[str], category: str = "") -> str:
    """Build the user prompt with concepts as JSON object."""
    domain = CATEGORY_CONTEXT.get(category, "")
    lines = [
        "---",
        content[:1500],
        "---",
    ]
    if domain:
        lines.append(f"Domain: {domain}")

    # Build concepts as JSON for structured input
    concept_dict = {}
    for name in concepts:
        plus_label, minus_label = CONCEPT_POLES[name]
        concept_dict[name] = {"plus": plus_label, "minus": minus_label}

    lines.append("Dimensions:")
    lines.append(json.dumps(concept_dict, ensure_ascii=False))

    return "\n".join(lines)


# ── Scoring via OpenAI-compatible API ────────────────────────

def _get_openai_client() -> Optional[Any]:
    """Create OpenAI client if API key is available."""
    try:
        import openai
    except ImportError:
        logger.warning("openai package not installed, cannot score stance")
        return None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.debug("No OPENAI_API_KEY, skipping stance scoring")
        return None

    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url

    return openai.OpenAI(**client_kwargs)


def _parse_json_response(raw: str) -> Optional[Dict]:
    """Strip markdown fences and parse JSON from nano response."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
    return json.loads(raw)


def _call_nano_batch(
    client: Any, model: str, content: str, concepts: List[str], category: str = "",
) -> Dict:
    """Call nano for one batch of concepts. Returns flat {concept: p_plus} dict."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": STANCE_SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(content, concepts, category)},
        ],
        temperature=0.0,
        max_tokens=1500,
    )
    raw = response.choices[0].message.content
    data = _parse_json_response(raw)
    return data if isinstance(data, dict) else {}


def score_stance(
    content: str, debug: bool = False,
) -> Optional[Dict[str, List[float]]]:
    """Score a memory's stance across all 48 concepts using nano.

    Sends one call per category (6 calls) so nano focuses on one
    topic domain at a time. Merges results before converting to lor arrays.

    Returns {"culture": [lor, lor, ...], "polity": [...], ...} or None on failure.
    When debug=True, result["_debug"] contains {concept: {p, conf, reason, lor, direction}}.
    """
    client = _get_openai_client()
    if client is None:
        return None

    model = os.environ.get("RESONANCE_MODEL", "gpt-4.1-nano")

    merged: Dict[str, Any] = {}
    try:
        for cat in CATEGORY_KEYS:
            concepts = LENS_CATEGORIES[cat]
            result = _call_nano_batch(client, model, content, concepts, category=cat)
            merged.update(result)
    except Exception:
        logger.exception("Stance scoring failed for content: %.80s", content)
        return None

    return _validate_and_convert(merged, debug=debug)


# Backward compat alias
score_resonance = score_stance


def _extract_stance(val: Any, concept_name: str) -> Optional[tuple]:
    """Extract (signed_strength, confidence, evidence, pole_chosen) from nano output.

    New format: {"e": "...", "d": "pole_name", "s": 0.0-1.0, "c": 0.0-1.0}
    d must match one of the two pole labels → determines sign.
    Returns (signed_s, conf, evidence, pole_chosen) or None.
    """
    if isinstance(val, (int, float)):
        # Legacy p_plus → convert to signed: s = 2*p - 1
        p = float(val)
        return (2.0 * p - 1.0, None, None, None)
    if not isinstance(val, dict):
        return None

    s = val.get("s")
    d = val.get("d", "")
    if s is None:
        return None
    try:
        s = float(s)
        s = max(0.0, min(1.0, s))
    except (TypeError, ValueError):
        return None

    conf = None
    try:
        if "c" in val and val["c"] is not None:
            conf = float(val["c"])
    except (TypeError, ValueError):
        pass

    evidence = str(val["e"]) if "e" in val else None

    # Determine sign from pole name
    is_minus = False
    if concept_name in CONCEPT_POLES:
        plus_label, minus_label = CONCEPT_POLES[concept_name]
        d_lower = str(d).lower().strip()
        plus_lower = plus_label.lower()
        minus_lower = minus_label.lower()

        # 1) Exact match (nano copied the label perfectly)
        if d_lower == minus_lower or d_lower == plus_lower:
            is_minus = (d_lower == minus_lower)
        # 2) Substring containment (d is part of a label, or label starts with d)
        elif d_lower in minus_lower and d_lower not in plus_lower:
            is_minus = True
        elif d_lower in plus_lower and d_lower not in minus_lower:
            is_minus = False
        # 3) Word overlap — only if one side clearly wins
        else:
            plus_words = set(plus_lower.replace(",", "").split())
            minus_words = set(minus_lower.replace(",", "").split())
            d_words = set(d_lower.replace(",", "").split())
            # Remove shared words (e.g., "emphasis", "orientation") — they don't help
            shared = plus_words & minus_words
            plus_only = len((d_words & plus_words) - shared)
            minus_only = len((d_words & minus_words) - shared)
            if minus_only > plus_only:
                is_minus = True
            elif plus_only > minus_only:
                is_minus = False
            else:
                # 4) Embedding fallback — cosine similarity
                result = _match_pole_by_embedding(d, plus_label, minus_label)
                if result is None:
                    logger.debug(
                        "Pole match ambiguous for %s: d=%r, discarding",
                        concept_name, d,
                    )
                    return None
                is_minus = result

        if is_minus:
            s = -s

    return (s, conf, evidence, str(d))


def s_to_lor(s: float) -> float:
    """Convert signed strength (-1..+1) to log-odds.

    Maps s to p = (s+1)/2, then logit(p).
    s=0 → lor=0, s=+0.9 → lor≈2.2, s=-0.9 → lor≈-2.2
    """
    p = (s + 1.0) / 2.0
    p = max(_P_EPS, min(1.0 - _P_EPS, p))
    return math.log(p / (1.0 - p))


def _validate_and_convert(
    data: Any, debug: bool = False,
) -> Optional[Dict[str, List[float]]]:
    """Validate sparse {concept: {e, s, c}} and convert to category lor arrays.

    Omitted concepts stay 0.0 (neutral). Only concepts with explicit
    stance values get converted to lor.
    When debug=True, result["_debug"] contains per-concept details.
    """
    if not isinstance(data, dict):
        return None

    # Initialize all categories with 0.0
    result: Dict[str, List[float]] = {}
    for cat in CATEGORY_KEYS:
        result[cat] = [0.0] * len(LENS_CATEGORIES[cat])

    debug_info: Dict[str, dict] = {}

    for concept_name, val in data.items():
        if concept_name not in CONCEPT_LOCATION:
            continue

        cat, idx = CONCEPT_LOCATION[concept_name]
        extracted = _extract_stance(val, concept_name)
        if extracted is None:
            continue

        s, conf, evidence, pole_chosen = extracted
        s = max(-1.0, min(1.0, s))
        lor = round(s_to_lor(s), 4)
        result[cat][idx] = lor

        if debug:
            plus_label, minus_label = CONCEPT_POLES[concept_name]
            debug_info[concept_name] = {
                "s": s,
                "conf": conf,
                "evidence": evidence,
                "pole": pole_chosen,
                "lor": lor,
                "direction": f"(+) {plus_label}" if s >= 0 else f"(-) {minus_label}",
            }

    if debug:
        result["_debug"] = debug_info  # type: ignore[assignment]
    return result


# ── FalkorDB write ───────────────────────────────────────────

def save_lor_to_graph(
    graph: Any,
    memory_id: str,
    lor_values: Dict[str, List[float]],
) -> bool:
    """Write lor arrays to Memory node in FalkorDB."""
    set_clauses = []
    params: Dict[str, Any] = {"mid": memory_id}

    for cat in CATEGORY_KEYS:
        if cat in lor_values:
            prop = lor_property_name(cat)
            param = "l_" + cat
            set_clauses.append("m.{} = ${}".format(prop, param))
            params[param] = lor_values[cat]

    if not set_clauses:
        return False

    query = "MATCH (m:Memory {id: $mid}) SET " + ", ".join(set_clauses)
    try:
        graph.query(query, params)
        return True
    except Exception:
        logger.exception("Failed to save lor for %s", memory_id)
        return False


# Backward compat alias
save_resonance_to_graph = save_lor_to_graph


# ── Batch scoring ────────────────────────────────────────────

def score_and_save(
    graph: Any,
    memory_id: str,
    content: str,
) -> Optional[Dict[str, List[float]]]:
    """Score stance and save to graph. Returns lor dict or None."""
    lors = score_stance(content)
    if lors is None:
        return None

    save_lor_to_graph(graph, memory_id, lors)
    return lors
