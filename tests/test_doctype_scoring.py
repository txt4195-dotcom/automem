"""Tests for doctype scoring pipeline stage."""

from __future__ import annotations

import math

import pytest

from automem.utils.doctype_scoring import (
    compute_doctype_score,
    compute_doctype_score_from_memory,
    parse_doctype_intent,
)
from automem.utils.lens_concepts import lor_concept_property


class TestParseDoctypeIntent:
    def test_basic_parse(self):
        result = parse_doctype_intent("decision:0.8,pattern:0.9")
        assert result == {
            "decision_emphasis": 0.8,
            "pattern_emphasis": 0.9,
        }

    def test_full_name(self):
        result = parse_doctype_intent("decision_emphasis:0.7")
        assert result == {"decision_emphasis": 0.7}

    def test_clamp_weight(self):
        result = parse_doctype_intent("decision:1.5,pattern:-0.3")
        assert result["decision_emphasis"] == 1.0
        assert result["pattern_emphasis"] == 0.0

    def test_empty_string(self):
        assert parse_doctype_intent("") == {}
        assert parse_doctype_intent("  ") == {}

    def test_invalid_format(self):
        assert parse_doctype_intent("gibberish") == {}
        assert parse_doctype_intent("unknown_key:0.5") == {}

    def test_mixed_valid_invalid(self):
        result = parse_doctype_intent("decision:0.8,bogus:0.5,insight:0.6")
        assert len(result) == 2
        assert "decision_emphasis" in result
        assert "insight_emphasis" in result

    def test_whitespace_tolerance(self):
        result = parse_doctype_intent(" decision : 0.8 , pattern : 0.9 ")
        assert result == {
            "decision_emphasis": 0.8,
            "pattern_emphasis": 0.9,
        }


class TestComputeDoctypeScore:
    def _sigmoid(self, x: float) -> float:
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        z = math.exp(x)
        return z / (1.0 + z)

    def test_strong_match(self):
        """Memory with high decision lor + query wants decisions → high score."""
        intent = {"decision_emphasis": 1.0}
        # lor=2.0 → sigmoid ≈ 0.88
        memory_lor = {"decision_emphasis": 2.0}
        score = compute_doctype_score(intent, memory_lor)
        assert score > 0.8

    def test_weak_match(self):
        """Memory with negative lor + query wants that type → low score."""
        intent = {"decision_emphasis": 1.0}
        memory_lor = {"decision_emphasis": -2.0}
        score = compute_doctype_score(intent, memory_lor)
        assert score < 0.2

    def test_neutral_lor(self):
        """lor=0 → sigmoid=0.5 → middling score."""
        intent = {"decision_emphasis": 1.0}
        memory_lor = {"decision_emphasis": 0.0}
        score = compute_doctype_score(intent, memory_lor)
        assert abs(score - 0.5) < 0.01

    def test_multi_dimension(self):
        """Multiple intent dimensions → averaged."""
        intent = {"decision_emphasis": 1.0, "pattern_emphasis": 1.0}
        memory_lor = {"decision_emphasis": 2.0, "pattern_emphasis": 2.0}
        score = compute_doctype_score(intent, memory_lor)
        expected = self._sigmoid(2.0)  # both same → average = same
        assert abs(score - expected) < 0.01

    def test_weight_matters(self):
        """Lower weight → lower contribution."""
        memory_lor = {"decision_emphasis": 2.0}
        score_high = compute_doctype_score({"decision_emphasis": 1.0}, memory_lor)
        score_low = compute_doctype_score({"decision_emphasis": 0.3}, memory_lor)
        assert score_high > score_low

    def test_empty_intent(self):
        memory_lor = {"decision_emphasis": 2.0, "pattern_emphasis": 1.0}
        assert compute_doctype_score({}, memory_lor) == 0.0

    def test_empty_lors(self):
        intent = {"decision_emphasis": 0.8}
        assert compute_doctype_score(intent, {}) == 0.0

    def test_none_inputs(self):
        assert compute_doctype_score(None, {"decision_emphasis": 1.0}) == 0.0
        assert compute_doctype_score({"decision_emphasis": 0.8}, None) == 0.0

    def test_zero_weight_skipped(self):
        """Weight=0 dimensions don't count."""
        intent = {"decision_emphasis": 0.0, "pattern_emphasis": 1.0}
        memory_lor = {"decision_emphasis": 2.0, "pattern_emphasis": -2.0}
        score = compute_doctype_score(intent, memory_lor)
        # Only pattern counted (lor=-2 → sigmoid ≈ 0.12)
        assert score < 0.2

    def test_missing_concept_in_lor(self):
        """Intent concept absent from memory_lor → contributes nothing."""
        intent = {"decision_emphasis": 1.0, "pattern_emphasis": 1.0}
        memory_lor = {"decision_emphasis": 2.0}  # pattern absent
        score_partial = compute_doctype_score(intent, memory_lor)
        score_full = compute_doctype_score({"decision_emphasis": 1.0}, memory_lor)
        # Partial has 2 dims but only 1 matched → normalized differently than 1-dim query
        # Both should be positive (decision matched), but full 1-dim = same numerator / 1
        # partial = same numerator / 1 (only 1 dim contributed) = same
        assert score_partial == score_full


class TestComputeDoctypeScoreFromMemory:
    def test_named_lor_properties(self):
        """v5: memory has named lor_doctype_decision_emphasis property."""
        memory = {
            lor_concept_property("decision_emphasis"): 2.0,
        }
        intent = {"decision_emphasis": 1.0}
        score = compute_doctype_score_from_memory(intent, memory)
        assert score > 0.8

    def test_nested_memory_format(self):
        """Recall results have memory dict nested under 'memory' key."""
        result = {
            "memory": {
                lor_concept_property("decision_emphasis"): 2.0,
            }
        }
        intent = {"decision_emphasis": 1.0}
        score = compute_doctype_score_from_memory(intent, result)
        assert score > 0.8

    def test_no_lor_doctype(self):
        memory = {"content": "hello"}
        intent = {"decision_emphasis": 1.0}
        assert compute_doctype_score_from_memory(intent, memory) == 0.0
