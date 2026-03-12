"""Tests for feedback-driven Beta lens update."""

import math

import pytest

from automem.utils.feedback_update import (
    VALID_SIGNALS,
    compute_lens_update,
    fetch_memory_lors,
    apply_lens_update,
)


# ── Helpers ──────────────────────────────────────────────────

def _uniform_lens(n=8):
    """Create a [1,1] uniform prior for n concepts."""
    return [[1.0, 1.0] for _ in range(n)]


def _make_lens(**kwargs):
    """Build a lens dict. kwargs: culture=[...], polity=[...], etc."""
    return kwargs


def _make_lors(**kwargs):
    """Build a lor dict. kwargs: culture=[...], polity=[...], etc."""
    return kwargs


# ── compute_lens_update ──────────────────────────────────────

class TestComputeLensUpdate:
    def test_helpful_moves_a_toward_p_node(self):
        """Positive signal: a should increase relative to p_node direction."""
        # Memory has strong positive stance on culture_0 (lor=2.0 → p_node≈0.88)
        lens = _make_lens(culture=[[1.0, 1.0], [1.0, 1.0]])
        lors = _make_lors(culture=[2.0, 0.0])  # strong stance on idx 0, neutral on idx 1

        updates = compute_lens_update(lens, lors, "helpful", top_k=5, min_stance=0.1)

        assert "culture_0" in updates
        # idx 1 has lor=0.0 → p_node=0.5 → stance=0.0 → below min_stance → skipped
        assert "culture_1" not in updates

        u = updates["culture_0"]
        a_new, b_new = u["after"]
        # For helpful: a gets more of the weight (p_node > 0.5)
        assert a_new > b_new, "a should be larger since p_node > 0.5"
        assert u["delta_p"] > 0, "p should increase toward the memory's stance"

    def test_opposite_view_reverses_direction(self):
        """Negative signal: should move away from memory's stance."""
        lens = _make_lens(culture=[[1.0, 1.0]])
        lors = _make_lors(culture=[2.0])  # p_node ≈ 0.88

        updates = compute_lens_update(lens, lors, "opposite_view", top_k=5, min_stance=0.1)

        assert "culture_0" in updates
        u = updates["culture_0"]
        # For opposite_view: b gets more weight (1-p_node applied to a)
        assert u["delta_p"] < 0, "p should decrease (away from memory's stance)"

    def test_not_relevant_returns_empty(self):
        lens = _make_lens(culture=[[1.0, 1.0]])
        lors = _make_lors(culture=[2.0])

        updates = compute_lens_update(lens, lors, "not_relevant")
        assert updates == {}

    def test_invalid_signal_returns_empty(self):
        lens = _make_lens(culture=[[1.0, 1.0]])
        lors = _make_lors(culture=[2.0])

        updates = compute_lens_update(lens, lors, "bogus_signal")
        assert updates == {}

    def test_novelty_damping_high_evidence_moves_less(self):
        """Established priors [50,50] should move much less than fresh [1,1]."""
        lors = _make_lors(culture=[2.0])

        fresh_lens = _make_lens(culture=[[1.0, 1.0]])
        established_lens = _make_lens(culture=[[50.0, 50.0]])

        fresh_updates = compute_lens_update(fresh_lens, lors, "helpful", top_k=5)
        established_updates = compute_lens_update(established_lens, lors, "helpful", top_k=5)

        fresh_delta = abs(fresh_updates["culture_0"]["delta_p"])
        established_delta = abs(established_updates["culture_0"]["delta_p"])

        assert fresh_delta > established_delta * 3, (
            f"Fresh prior should move much more: {fresh_delta} vs {established_delta}"
        )

    def test_top_k_filtering(self):
        """Only top-K axes by stance should be updated."""
        # 5 concepts with decreasing stance
        lors_vals = [3.0, 2.0, 1.0, 0.5, 0.3]
        lens = _make_lens(culture=[[1.0, 1.0] for _ in range(5)])
        lors = _make_lors(culture=lors_vals)

        updates = compute_lens_update(lens, lors, "helpful", top_k=3, min_stance=0.1)

        # Should only have top 3
        assert len(updates) == 3
        # The strongest stances should be selected
        assert "culture_0" in updates  # lor=3.0
        assert "culture_1" in updates  # lor=2.0
        assert "culture_2" in updates  # lor=1.0

    def test_min_stance_filters_weak_signals(self):
        """Axes with stance below min_stance should be skipped."""
        lens = _make_lens(culture=[[1.0, 1.0], [1.0, 1.0]])
        lors = _make_lors(culture=[0.01, 2.0])  # 0.01 → nearly neutral

        updates = compute_lens_update(lens, lors, "helpful", top_k=5, min_stance=0.1)

        assert "culture_0" not in updates, "Near-neutral axis should be filtered"
        assert "culture_1" in updates

    def test_empty_inputs(self):
        assert compute_lens_update({}, {}, "helpful") == {}
        assert compute_lens_update(None, None, "helpful") == {}
        assert compute_lens_update({"culture": [[1, 1]]}, {}, "helpful") == {}
        assert compute_lens_update({}, {"culture": [2.0]}, "helpful") == {}

    def test_multi_category(self):
        """Updates can span multiple categories."""
        lens = _make_lens(
            culture=[[1.0, 1.0]],
            polity=[[1.0, 1.0]],
        )
        lors = _make_lors(
            culture=[2.0],
            polity=[-1.5],
        )

        updates = compute_lens_update(lens, lors, "helpful", top_k=10, min_stance=0.1)

        assert "culture_0" in updates
        assert "polity_0" in updates

    def test_update_preserves_evidence_direction(self):
        """After helpful feedback on positive lor, a/(a+b) should increase."""
        lens = _make_lens(culture=[[5.0, 5.0]])  # p=0.5 initially
        lors = _make_lors(culture=[2.0])  # p_node ≈ 0.88

        updates = compute_lens_update(lens, lors, "helpful", top_k=5)
        u = updates["culture_0"]

        a_new, b_new = u["after"]
        p_new = a_new / (a_new + b_new)
        assert p_new > 0.5, f"Should move toward p_node direction: p={p_new}"


# ── fetch_memory_lors ────────────────────────────────────────

class TestFetchMemoryLors:
    def test_returns_none_on_no_result(self):
        class EmptyGraph:
            def query(self, q, p):
                class R:
                    result_set = []
                return R()

        assert fetch_memory_lors(EmptyGraph(), "mem-1") is None

    def test_returns_lors_dict(self):
        class MockGraph:
            def query(self, q, p):
                # Returns 7 values (one per category including doctype)
                class R:
                    result_set = [
                        [[1.0, -0.5], None, None, None, None, None, None]
                    ]
                return R()

        result = fetch_memory_lors(MockGraph(), "mem-1")
        assert result is not None
        assert "culture" in result
        assert result["culture"] == [1.0, -0.5]


# ── apply_lens_update ────────────────────────────────────────

class TestApplyLensUpdate:
    def test_writes_updated_values(self):
        """Verify graph SET is called with correct updated values."""
        queries = []

        class MockGraph:
            def query(self, q, p):
                queries.append((q, p))
                if "RETURN" in q:
                    class R:
                        result_set = [
                            [[[1.0, 1.0], [2.0, 3.0]]]
                        ]
                    return R()
                class R:
                    result_set = []
                return R()

        updates = {
            "culture_0": {
                "cat": "culture",
                "idx": 0,
                "before": [1.0, 1.0],
                "after": [1.5, 1.2],
                "delta_p": 0.05,
            },
        }

        result = apply_lens_update(MockGraph(), "user:test", updates)
        assert result is True

        # Should have a fetch and a set query
        assert len(queries) == 2
        set_query, set_params = queries[1]
        assert "SET" in set_query
        assert set_params["val"][0] == [1.5, 1.2]  # updated
        assert set_params["val"][1] == [2.0, 3.0]   # unchanged

    def test_returns_true_on_empty_updates(self):
        assert apply_lens_update(None, "user:test", {}) is True
