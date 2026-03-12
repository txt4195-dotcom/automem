"""Generate scenario-based derived content from enrichment analysis.

Takes the structured enrichment JSON (from reasoning_generator v7) and
produces concrete scenarios where someone would need this content but
would search using completely different words. Each scenario becomes
an independent Memory node linked via DERIVED_FROM.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SCENARIO_SYSTEM_PROMPT = """\
You generate concrete usage scenarios for a piece of content.
Each scenario describes a real situation where someone would need
this content but would search using completely different words.

You receive the original content and its structured analysis
(overstory, inference, frame, limits, bridges, biases).

Generate 5 scenarios. For each scenario output a JSON object with:
- "scenario": what is happening (1 sentence)
- "role": who is the person (job title or archetype)
- "context": why they need this content (1-2 sentences)
- "search_words": the words they would actually type to search
  (must be DIFFERENT vocabulary from the original content)

Output as a JSON array of 5 scenario objects."""

SCENARIO_USER_TEMPLATE = """\
CONTENT:
{content}

ANALYSIS:
{analysis}"""


def generate_scenarios(
    content: str,
    enrichment_json: Dict[str, Any],
) -> Optional[List[Dict[str, str]]]:
    """Generate scenario derived content from enrichment analysis.

    Returns a list of scenario dicts, or None on failure.
    Each dict has: scenario, role, context, search_words.
    """
    if not content or len(content.strip()) < 20:
        return None

    from automem.utils.reasoning_generator import _get_openai_client

    client = _get_openai_client()
    if client is None:
        return None

    model = os.environ.get("REASONING_MODEL", "gpt-5-nano")
    reasoning_effort = os.environ.get("SCENARIO_REASONING_EFFORT",
                                      os.environ.get("REASONING_EFFORT", "medium"))
    max_tokens = int(os.environ.get("SCENARIO_MAX_TOKENS", "8000"))

    analysis_text = json.dumps(enrichment_json, ensure_ascii=False, indent=None)
    # Truncate analysis if too long
    if len(analysis_text) > 3000:
        analysis_text = analysis_text[:3000]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SCENARIO_SYSTEM_PROMPT},
                {"role": "user", "content": SCENARIO_USER_TEMPLATE.format(
                    content=content[:2000],
                    analysis=analysis_text,
                )},
            ],
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_tokens,
        )
        raw = response.choices[0].message.content
        if not raw:
            logger.debug("Empty scenario response for content: %.60s", content)
            return None

        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        try:
            scenarios = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            logger.debug("Scenario JSON parse failed for content: %.60s", content)
            return None

        if not isinstance(scenarios, list):
            logger.debug("Scenario response is not a list for content: %.60s", content)
            return None

        valid = []
        for s in scenarios:
            if isinstance(s, dict) and s.get("scenario"):
                valid.append({
                    "scenario": s.get("scenario", ""),
                    "role": s.get("role", ""),
                    "context": s.get("context", ""),
                    "search_words": s.get("search_words", ""),
                })

        if valid:
            logger.debug("Generated %d scenarios for content: %.60s", len(valid), content)
            return valid
        return None

    except Exception:
        logger.exception("Scenario generation failed for content: %.80s", content)
        return None


def build_scenario_content(scenario: Dict[str, str]) -> str:
    """Build content text for a scenario Memory node.

    Includes all fields so the embedding captures the full search surface.
    """
    parts = []
    if scenario.get("scenario"):
        parts.append(scenario["scenario"])
    if scenario.get("role"):
        parts.append(f"Role: {scenario['role']}")
    if scenario.get("context"):
        parts.append(scenario["context"])
    if scenario.get("search_words"):
        parts.append(scenario["search_words"])
    return " ".join(parts)
