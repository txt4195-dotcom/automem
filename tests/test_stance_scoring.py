"""Tests for the stance scoring → lor → profile alignment pipeline.

No real nano calls — tests the math pipeline only:
  _extract_stance (pole matching) → s_to_lor → compute_profile_score
"""

import math

import pytest

from automem.utils.resonance_scorer import (
    _extract_stance,
    _validate_and_convert,
    p_to_lor,
    s_to_lor,
)
from automem.utils.user_profile import (
    compute_profile_score,
    get_memory_resonance,
)
from automem.utils.lens_concepts import (
    CATEGORY_KEYS,
    LENS_CATEGORIES,
    lor_concept_property,
)


# ── s_to_lor conversion ──────────────────────────────────────

class TestSToLor:
    def test_neutral(self):
        assert abs(s_to_lor(0.0)) < 0.01

    def test_positive(self):
        lor = s_to_lor(0.9)
        assert lor > 2.0  # strong plus

    def test_negative(self):
        lor = s_to_lor(-0.9)
        assert lor < -2.0  # strong minus

    def test_symmetry(self):
        assert abs(s_to_lor(0.7) + s_to_lor(-0.7)) < 0.01


# ── _extract_stance (pole matching) ──────────────────────────

class TestExtractStance:
    def test_exact_match_plus(self):
        """nano copies plus label exactly."""
        val = {"e": "I love freedom", "d": "individualism, personal autonomy", "s": 0.8, "c": 0.9}
        result = _extract_stance(val, "individualism_collectivism")
        assert result is not None
        s, conf, evidence, pole = result
        assert s > 0  # plus pole → positive
        assert abs(s - 0.8) < 0.01

    def test_exact_match_minus(self):
        """nano copies minus label exactly."""
        val = {"e": "group harmony matters", "d": "collectivism, group harmony", "s": 0.7, "c": 0.8}
        result = _extract_stance(val, "individualism_collectivism")
        assert result is not None
        s, conf, evidence, pole = result
        assert s < 0  # minus pole → negative
        assert abs(s - (-0.7)) < 0.01

    def test_substring_match(self):
        """nano writes partial label — substring match."""
        val = {"e": "quote", "d": "collectivism", "s": 0.6, "c": 0.7}
        result = _extract_stance(val, "individualism_collectivism")
        assert result is not None
        s, _, _, _ = result
        assert s < 0  # "collectivism" → minus pole

    def test_word_overlap_match(self):
        """nano writes fuzzy label — word overlap match."""
        val = {"e": "quote", "d": "personal autonomy", "s": 0.5, "c": 0.6}
        result = _extract_stance(val, "individualism_collectivism")
        assert result is not None
        s, _, _, _ = result
        assert s > 0  # "personal autonomy" has unique overlap with plus

    def test_missing_s(self):
        """No strength → returns None."""
        val = {"e": "quote", "d": "something"}
        assert _extract_stance(val, "individualism_collectivism") is None

    def test_legacy_float(self):
        """Legacy p_plus format (bare float) → converts to signed."""
        result = _extract_stance(0.9, "individualism_collectivism")
        assert result is not None
        s, _, _, _ = result
        assert abs(s - 0.8) < 0.01  # 2*0.9 - 1 = 0.8

    def test_clamps_strength(self):
        """s outside 0-1 gets clamped."""
        val = {"e": "quote", "d": "individualism, personal autonomy", "s": 1.5, "c": 0.5}
        result = _extract_stance(val, "individualism_collectivism")
        assert result is not None
        s, _, _, _ = result
        assert abs(s - 1.0) < 0.01  # clamped to 1.0


# ── _validate_and_convert (nano output → concept lor dict) ───

class TestValidateAndConvert:
    def test_sparse_to_named(self):
        """Sparse nano output returns {concept_name: lor_float} — only scored concepts."""
        data = {
            "individualism_collectivism": {
                "e": "self-reliance emphasized",
                "d": "individualism, personal autonomy",
                "s": 0.8,
                "c": 0.9,
            },
        }
        result = _validate_and_convert(data)
        assert result is not None
        assert "individualism_collectivism" in result
        assert result["individualism_collectivism"] > 0  # plus pole
        # Other concepts not scored → absent from result
        assert "hierarchy_egalitarian" not in result

    def test_debug_mode(self):
        data = {
            "certainty_ambiguity": {
                "e": "embrace the unknown",
                "d": "ambiguity, comfort with not knowing",
                "s": 0.9,
                "c": 0.9,
            },
        }
        result = _validate_and_convert(data, debug=True)
        assert result is not None
        assert "_debug" in result
        debug = result["_debug"]
        assert "certainty_ambiguity" in debug
        assert debug["certainty_ambiguity"]["lor"] < 0  # minus pole

    def test_unknown_concept_ignored(self):
        data = {"nonexistent_concept": {"e": "x", "d": "y", "s": 0.5, "c": 0.5}}
        result = _validate_and_convert(data)
        assert result is not None
        # No real concepts scored → only _debug could be present, but no concept keys
        concept_keys = [k for k in result if not k.startswith("_")]
        assert len(concept_keys) == 0

    def test_empty_dict(self):
        result = _validate_and_convert({})
        assert result is not None  # valid but all zeros


# ── compute_profile_score (alignment) ────────────────────────

class TestProfileScore:
    # compute_profile_score(lens, content_lor) expects both dicts keyed by
    # concept_name (e.g. "individualism_collectivism"), not property names.
    # lens: {concept_name: [a, b]}
    # content_lor: {concept_name: lor_float}

    def _make_simple_lens(self, a, b):
        """Create lens with one concept having Beta(a, b)."""
        return {"individualism_collectivism": [a, b]}

    def _make_simple_lor(self, lor_value):
        """Create lor dict with individualism_collectivism = lor_value."""
        return {"individualism_collectivism": lor_value}

    def test_aligned_positive(self):
        """User leans individualism, memory leans individualism → positive score."""
        lens = self._make_simple_lens(8, 2)  # p_user=0.8 → strong individualism
        lors = self._make_simple_lor(2.0)    # lor>0 → individualism
        score = compute_profile_score(lens, lors)
        assert score > 0

    def test_aligned_negative(self):
        """User leans collectivism, memory leans collectivism → positive score."""
        lens = self._make_simple_lens(2, 8)  # p_user=0.2 → strong collectivism
        lors = self._make_simple_lor(-2.0)   # lor<0 → collectivism
        score = compute_profile_score(lens, lors)
        assert score > 0

    def test_misaligned(self):
        """User leans individualism, memory leans collectivism → negative score."""
        lens = self._make_simple_lens(8, 2)  # strong individualism
        lors = self._make_simple_lor(-2.0)   # collectivism
        score = compute_profile_score(lens, lors)
        assert score < 0

    def test_neutral_content_ignored(self):
        """Content lor near 0 → skipped (no contribution)."""
        lens = self._make_simple_lens(8, 2)
        lors = self._make_simple_lor(0.001)  # below _MIN_LOR threshold
        score = compute_profile_score(lens, lors)
        assert score == 0.0

    def test_uniform_prior_ignored(self):
        """User has [1,1] uniform prior → skipped (no opinion)."""
        lens = self._make_simple_lens(1, 1)
        lors = self._make_simple_lor(2.0)
        score = compute_profile_score(lens, lors)
        assert score == 0.0

    def test_empty_lens(self):
        assert compute_profile_score({}, {"individualism_collectivism": 2.0}) == 0.0

    def test_empty_lor(self):
        lens = self._make_simple_lens(8, 2)
        assert compute_profile_score(lens, {}) == 0.0

    def test_stronger_alignment_higher_score(self):
        """Stronger content stance → higher absolute score."""
        lens = self._make_simple_lens(8, 2)
        weak = compute_profile_score(lens, self._make_simple_lor(0.5))
        strong = compute_profile_score(lens, self._make_simple_lor(3.0))
        assert strong > weak > 0

    def test_multi_category(self):
        """Score sums across multiple categories."""
        multi_lens = {
            "individualism_collectivism": [8, 2],
            "empiricism_revelation": [8, 2],
        }
        multi_lors = {
            "individualism_collectivism": 2.0,
            "empiricism_revelation": 2.0,
        }
        single_lens = {"individualism_collectivism": [8, 2]}
        single_lors = {"individualism_collectivism": 2.0}

        multi_score = compute_profile_score(multi_lens, multi_lors)
        single_score = compute_profile_score(single_lens, single_lors)
        # Multi should be roughly 2x single (both aligned)
        assert multi_score > single_score
        assert abs(multi_score - 2 * single_score) < 0.01


# ── get_memory_resonance (extraction from memory dict) ───────

class TestGetMemoryResonance:
    def test_extracts_named_lor_properties(self):
        """v5: reads lor_culture_individualism_collectivism, etc."""
        memory = {
            lor_concept_property("individualism_collectivism"): 0.5,
            lor_concept_property("hierarchy_egalitarian"): -0.3,
        }
        resonance = get_memory_resonance(memory)
        assert resonance is not None
        assert "individualism_collectivism" in resonance
        assert resonance["individualism_collectivism"] == 0.5
        assert resonance["hierarchy_egalitarian"] == -0.3

    def test_returns_none_when_no_lor(self):
        memory = {"content": "hello", "tags": ["test"]}
        assert get_memory_resonance(memory) is None

    def test_skips_zero_values(self):
        """lor=0.0 is neutral and should not appear in resonance."""
        memory = {
            lor_concept_property("individualism_collectivism"): 0.0,
        }
        assert get_memory_resonance(memory) is None


# ── End-to-end: nano output → lor → alignment ────────────────

class TestEndToEnd:
    def test_full_pipeline(self):
        """Simulate: nano scores content → convert to lor → score against user lens."""
        # 1. Simulated nano output for a strongly individualistic text
        nano_output = {
            "individualism_collectivism": {
                "e": "self-reliance and personal freedom",
                "d": "individualism, personal autonomy",
                "s": 0.9,
                "c": 0.9,
            },
            "consumerism_minimalism": {
                "e": "less is more, simplify life",
                "d": "minimalism, less is more",
                "s": 0.8,
                "c": 0.8,
            },
        }

        # 2. Convert nano output to named concept lor dict
        lor_result = _validate_and_convert(nano_output)
        assert lor_result is not None

        # individualism → plus pole → positive lor
        assert lor_result["individualism_collectivism"] > 0
        # minimalism → minus pole → negative lor
        assert lor_result["consumerism_minimalism"] < 0

        # 3. Create user lens: also individualist, also minimalist
        # compute_profile_score accepts concept_name keys (v5 named format)
        user_lens = {
            "individualism_collectivism": [8, 2],   # p_user=0.8 → individualism
            "consumerism_minimalism": [2, 8],        # p_user=0.2 → minimalism (minus pole)
        }

        # 4. Compute profile score
        score = compute_profile_score(user_lens, lor_result)
        # Both dimensions aligned → should be positive
        assert score > 0

        # 5. Now test misalignment: user is collectivist/consumerist
        user_lens_opposite = {
            "individualism_collectivism": [2, 8],  # collectivism
            "consumerism_minimalism": [8, 2],       # consumerism
        }

        score_opposite = compute_profile_score(user_lens_opposite, lor_result)
        # Both dimensions misaligned → should be negative
        assert score_opposite < 0

        # Aligned score should be higher than misaligned
        assert score > score_opposite
