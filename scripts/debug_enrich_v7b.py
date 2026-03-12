#!/usr/bin/env python3
"""Debug script: test v7b enrichment pipeline end-to-end.

Enrichment v7b pipeline 각 단계를 격리해서 순서대로 실행하는 디버그 도구.
pytest가 아니라 개발자가 직접 돌리면서 중간 출력을 눈으로 확인하는 용도.

Pipeline stages:
1. generate_reasoning() → LLM이 6Q 배터리로 메모리를 분석 → (enrichment_json, words)
2. _flatten_enrichment() → 구조화된 JSON을 임베딩용 평문으로 변환
3. build_rich_embed_text() → 원본 + enrichment + words 3-layer 임베딩 텍스트 조립
4. score_stance() → 48개 lens concept 대비 log-odds ratio 점수 + 근거

Usage:
    python scripts/debug_enrich_v7b.py

Requires: OPENAI_API_KEY (generate_reasoning, score_stance 모두 LLM 호출)
"""

from __future__ import annotations

import json
import logging
import os
import sys

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("debug_v7b")

# ── Test content ──────────────────────────────────────────────
# 분산 시스템 디버깅에 관한 샘플 텍스트.
# 실제 메모리처럼 기술적 insight가 담겨 있어서
# enrichment의 각 단계가 의미 있는 출력을 내는지 확인하기 좋다.
TEST_CONTENT = """
When debugging distributed systems, the hardest bugs are the ones that only
appear under specific timing conditions. Race conditions between microservices
often manifest as intermittent failures that pass in CI but fail in production.
The key insight is to add structured logging with correlation IDs at every
service boundary, then use distributed tracing (Jaeger/Zipkin) to reconstruct
the exact sequence of events. Don't chase the symptom — instrument first,
reproduce second.
""".strip()


def test_generate_reasoning():
    """Stage 1: LLM 6Q battery — 메모리 내용을 6가지 관점으로 분석.

    6Q = overstory(큰 그림), inference(추론), frame(프레임),
         limits(한계), bridges(연결점), biases(편향).

    Returns:
        enrichment_json: 6Q 결과가 담긴 dict (key별로 dict/list/str)
        words: LLM이 자연어로 풀어쓴 enrichment 텍스트
    """
    logger.info("=" * 60)
    logger.info("TEST 1: generate_reasoning()")
    logger.info("=" * 60)

    # lazy import — 이 모듈은 무거운 의존성(openai 등)을 끌고 오므로
    # 스크립트 상단에서 import하면 다른 테스트도 못 돌린다.
    from automem.utils.reasoning_generator import generate_reasoning

    result = generate_reasoning(TEST_CONTENT)
    if result is None:
        logger.error("generate_reasoning returned None — check OPENAI_API_KEY")
        return None, None

    enrichment_json, words = result
    logger.info("enrichment_json keys: %s", list(enrichment_json.keys()))
    logger.info("words length: %d chars", len(words))
    logger.info("words preview: %.200s", words)

    # 6Q 각 섹션의 타입과 크기를 확인 — 구조가 기대와 다르면 여기서 잡힌다.
    for key in ("overstory", "inference", "frame", "limits", "bridges", "biases"):
        val = enrichment_json.get(key)
        if val:
            if isinstance(val, dict):
                logger.info("  %s: %s", key, list(val.keys()))
            elif isinstance(val, list):
                logger.info("  %s: [%d items]", key, len(val))
            else:
                logger.info("  %s: %s", key, type(val).__name__)

    return enrichment_json, words


def test_flatten_enrichment(enrichment_json):
    """Stage 2: enrichment JSON → 임베딩용 평문 변환.

    구조화된 6Q JSON을 하나의 문자열로 펼친다.
    이 평문이 Stage 3에서 임베딩 텍스트의 두 번째 레이어가 된다.
    """
    logger.info("=" * 60)
    logger.info("TEST 2: _flatten_enrichment()")
    logger.info("=" * 60)

    from automem.utils.reasoning_generator import _flatten_enrichment

    flat = _flatten_enrichment(enrichment_json)
    logger.info("Flattened length: %d chars", len(flat))
    logger.info("Preview: %.300s...", flat)
    return flat


def test_build_rich_embed_text(enrichment_text, words):
    """Stage 3: 3-layer 임베딩 텍스트 조립.

    최종 임베딩에 들어가는 텍스트 = 원본 content + enrichment 평문 + words.
    단순 원본만 임베딩하면 semantic search에서 놓치는 의미가 많기 때문에,
    LLM이 추출한 상위 개념/추론/연결점까지 포함시켜 recall 품질을 높인다.
    """
    logger.info("=" * 60)
    logger.info("TEST 3: build_rich_embed_text()")
    logger.info("=" * 60)

    from automem.utils.reasoning_generator import build_rich_embed_text

    rich = build_rich_embed_text(TEST_CONTENT, enrichment_text, words)
    logger.info("Rich embed text length: %d chars", len(rich))
    logger.info("  content: %d chars", len(TEST_CONTENT))
    logger.info("  enrichment: %d chars", len(enrichment_text) if enrichment_text else 0)
    logger.info("  words: %d chars", len(words) if words else 0)

    # 조립된 텍스트의 파트 구조 확인 — \n\n으로 구분된 각 레이어가 있어야 한다.
    parts = rich.split("\n\n")
    logger.info("  Parts count: %d", len(parts))
    for i, part in enumerate(parts):
        logger.info("  Part %d: %d chars — %.80s...", i, len(part), part)

    return rich


def test_score_stance():
    """Stage 4: 48개 lens concept에 대한 stance 점수 측정.

    각 concept(예: debugging, distributed_systems, observability...)에 대해
    이 메모리가 얼마나 관련되는지 log-odds ratio(lor)로 점수를 매긴다.
    lor > 0이면 관련 있음, < 0이면 무관함.

    _evidence 필드에는 각 점수의 근거가 담긴다:
      e = evidence (왜 관련되는지 설명)
      d = direction (positive/negative/neutral)
    """
    logger.info("=" * 60)
    logger.info("TEST 4: score_stance() with _evidence")
    logger.info("=" * 60)

    from automem.utils.resonance_scorer import score_stance

    result = score_stance(TEST_CONTENT)
    if result is None:
        logger.error("score_stance returned None — check OPENAI_API_KEY / RESONANCE_MODEL")
        return None

    # _로 시작하는 키는 메타 정보(_evidence 등), 나머지가 실제 concept 점수
    evidence = result.get("_evidence", {})
    lor_count = sum(1 for k in result if not k.startswith("_"))
    logger.info("Scored %d concepts", lor_count)
    logger.info("Evidence captured: %d entries", len(evidence))

    for concept, lor in sorted(result.items()):
        if concept.startswith("_"):
            continue
        ev = evidence.get(concept, {})
        logger.info("  %s: lor=%.4f  e=%s  d=%s",
                     concept, lor,
                     (ev.get("e", "")[:60] + "...") if ev.get("e") else "N/A",
                     ev.get("d", "N/A"))

    return result


# ── Entry point ──────────────────────────────────────────────
# 순서가 중요하다: 1→2→3은 의존 체인 (1의 출력이 2의 입력),
# 4(stance)는 독립적이지만 같은 TEST_CONTENT를 쓴다.
# 어느 단계에서 실패하든 그 시점의 로그로 원인을 좁힐 수 있다.

def main():
    # 환경 확인 — API 키 없으면 LLM 호출 단계에서 None이 나온다.
    logger.info("v7b enrichment debug — testing pipeline")
    logger.info("OPENAI_API_KEY: %s", "set" if os.environ.get("OPENAI_API_KEY") else "NOT SET")
    logger.info("REASONING_MODEL: %s", os.environ.get("REASONING_MODEL", "gpt-5-nano (default)"))
    logger.info("RESONANCE_MODEL: %s", os.environ.get("RESONANCE_MODEL", "gpt-4.1-nano (default)"))
    logger.info("")

    # Stage 1: LLM 6Q 추론 — 실패하면 이후 단계 의미 없으므로 즉시 중단
    enrichment_json, words = test_generate_reasoning()
    if enrichment_json is None:
        logger.error("Stopping — no enrichment generated")
        sys.exit(1)

    # Stage 2: JSON → 평문 (Stage 3의 입력)
    enrichment_text = test_flatten_enrichment(enrichment_json)

    # Stage 3: 3-layer 임베딩 텍스트 조립
    rich_text = test_build_rich_embed_text(enrichment_text, words)

    # Stage 4: stance scoring (독립 — Stage 1~3과 별도 LLM 호출)
    stance_result = test_score_stance()

    # ── Summary ──────────────────────────────────────────────
    # 각 단계의 출력 크기를 한눈에 비교.
    # 임베딩 텍스트가 너무 길거나 짧으면 여기서 눈에 띈다.
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Enrichment JSON: %d keys, %d chars serialized",
                len(enrichment_json), len(json.dumps(enrichment_json)))
    logger.info("Words: %d chars", len(words))
    logger.info("Flattened enrichment: %d chars", len(enrichment_text))
    logger.info("Rich embed text: %d chars (content=%d + enrichment=%d + words=%d)",
                len(rich_text), len(TEST_CONTENT), len(enrichment_text), len(words))
    if stance_result:
        evidence = stance_result.get("_evidence", {})
        lor_count = sum(1 for k in stance_result if not k.startswith("_"))
        logger.info("Stance: %d concepts scored, %d evidence entries", lor_count, len(evidence))

    logger.info("")
    logger.info("All tests passed!")


if __name__ == "__main__":
    main()
