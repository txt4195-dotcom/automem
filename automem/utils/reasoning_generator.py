"""Generate structured enrichment for memory content (v7b).

Uses gpt-5-nano (reasoning model) with a 6-question battery to produce
structured JSON. The full enrichment text (descriptions + words) is embedded
alongside content in Qdrant vectors. The structured JSON goes into Qdrant
payload. FalkorDB stores only the words text (backward compat).

Questions:
  1. Overstory — what larger narrative is this one chapter of
  2. Inference — reasoning path with intermediate concepts
  3. Frame — intersubjective reality that makes this meaningful
  4. Limits — when this breaks or becomes wrong
  5. Bridges — equivalent concepts in distant domains
  6. Cognitive biases — what biases distort judgment about this content
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

ENRICHMENT_SYSTEM_PROMPT = """\
You analyze content and generate structured enrichment for vector search.
Your output will be stored as structured data. The "words" field in each
answer will be embedded alongside the original content for search — every
term you put there becomes a searchable surface in vector space.

Answer these questions about the input.
For EVERY answer, include a "words" field — a flat string of 10-20
related terms, synonyms, translations, jargon variants, and associated
concepts that someone might use when searching for this content.
These words are for vector embedding, not for humans.

1. OVERSTORY
What larger narrative is this content one chapter of?
Not the backstory or context — the overarching story that connects
this to other instances of the same pattern across domains and time.
Name the overstory explicitly.

2. INFERENCE
What reasoning path produced this content?
Lay out the chain: premise → intermediate concept → conclusion.
Name each intermediate concept — these are the search bridges
that connect this content to queries using different vocabulary.

3. FRAME
What intersubjective reality makes this content meaningful?
What must a community collectively believe for this to be "true"?
Name the frame. Then: what happens when viewed from outside it?
What alternative frame would make this content wrong or irrelevant?

4. LIMITS
When does this content break, become wrong, or become dangerous to apply?
What conditions must hold for it to be valid?
What's the most common mistake when applying this?

5. BRIDGES
Name equivalent concepts in 3 domains far from the original.
For each: domain, concept name, and one sentence explaining
why it's structurally the same pattern.

6. COGNITIVE BIASES
What cognitive biases activate when someone encounters this content?
Both biases that make people over-accept it (e.g. confirmation bias,
authority bias) and biases that make people reject it (e.g. status quo
bias, loss aversion). Name each bias and explain how it distorts
judgment about this specific content.

Output as JSON with keys: overstory, inference, frame, limits, bridges, biases.
Every object must include a "words" field."""

ENRICHMENT_USER_TEMPLATE = """\
---
{content}
---"""

ENRICHMENT_KEYS = ("overstory", "inference", "frame", "limits", "bridges", "biases")


def _get_openai_client() -> Optional[Any]:
    """Create OpenAI client if API key is available."""
    try:
        import openai
    except ImportError:
        logger.warning("openai package not installed, cannot generate enrichment")
        return None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.debug("No OPENAI_API_KEY, skipping enrichment generation")
        return None

    kwargs: dict = {"api_key": api_key}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return openai.OpenAI(**kwargs)


def _extract_all_words(data: Dict[str, Any]) -> str:
    """Extract all 'words' fields from enrichment JSON.

    Returns a single string of related terms for vector embedding.
    """
    words = []
    for key in ENRICHMENT_KEYS:
        val = data.get(key)
        if isinstance(val, dict):
            w = val.get("words", "")
            if w:
                words.append(w)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    w = item.get("words", "")
                    if w:
                        words.append(w)
        elif isinstance(val, str):
            # Some models return plain strings — no words to extract
            pass
    return " ".join(words)


def _flatten_enrichment(data: Dict[str, Any]) -> str:
    """Flatten enrichment JSON into text for vector embedding.

    Includes ALL text from every section (descriptions + words).
    Maximizes the search surface in Qdrant vector space.
    """
    parts = []
    for key in ENRICHMENT_KEYS:
        val = data.get(key)
        if isinstance(val, dict):
            for v in val.values():
                if isinstance(v, str) and v:
                    parts.append(v)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            parts.extend(str(x) for x in item.values() if x)
                        elif isinstance(item, str) and item:
                            parts.append(item)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    parts.extend(str(v) for v in item.values() if v)
                elif isinstance(item, str) and item:
                    parts.append(item)
        elif isinstance(val, str) and val:
            parts.append(val)
    return " ".join(parts)


def _parse_llm_json(raw: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None


def generate_reasoning(content: str) -> Optional[Tuple[Dict[str, Any], str]]:
    """Generate structured enrichment for a memory's content.

    Uses gpt-5-nano (reasoning model) with 6-question battery.
    Returns (enrichment_json, words_text) tuple, or None on failure.

    - enrichment_json: full structured analysis (for Qdrant payload + vector via _flatten_enrichment)
    - words_text: extracted keywords only (for FalkorDB reasoning field, backward compat)
    """
    if not content or len(content.strip()) < 20:
        return None

    client = _get_openai_client()
    if client is None:
        return None

    model = os.environ.get("REASONING_MODEL", "gpt-5-nano")
    reasoning_effort = os.environ.get("REASONING_EFFORT", "high")
    max_tokens = int(os.environ.get("REASONING_MAX_TOKENS", "16000"))

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ENRICHMENT_SYSTEM_PROMPT},
                {"role": "user", "content": ENRICHMENT_USER_TEMPLATE.format(
                    content=content[:3000]
                )},
            ],
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_tokens,
        )
        raw = response.choices[0].message.content
        if not raw:
            logger.debug("Empty enrichment response for content: %.60s", content)
            return None

        data = _parse_llm_json(raw)
        if data is None:
            logger.debug("Enrichment JSON parse failed, using raw text as words")
            return ({"_raw": raw.strip()}, raw.strip())

        words = _extract_all_words(data)

        usage = response.usage
        reasoning_tokens = 0
        if usage and hasattr(usage, "completion_tokens_details"):
            d = usage.completion_tokens_details
            if d and hasattr(d, "reasoning_tokens"):
                reasoning_tokens = d.reasoning_tokens
        logger.debug(
            "Generated enrichment (%d words chars, %d reasoning tokens) for: %.60s",
            len(words), reasoning_tokens, content,
        )
        return (data, words) if words else (data, "")
    except Exception:
        logger.exception("Enrichment generation failed for content: %.80s", content)
        return None


def build_rich_embed_text(
    content: str,
    enrichment_text: Optional[str] = None,
    words: Optional[str] = None,
) -> str:
    """Build the text to embed in Qdrant.

    content + enrichment descriptions + words = rich vector.
    All three layers maximize the search surface:
    - content: original text
    - enrichment_text: flattened analysis (overstory, inference, frame, etc.)
    - words: extracted related terms and synonyms
    """
    parts = [content]
    if enrichment_text:
        parts.append(enrichment_text)
    if words:
        parts.append(words)
    return "\n\n".join(parts)
