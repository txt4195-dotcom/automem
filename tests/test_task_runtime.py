from __future__ import annotations

from automem.search.task_runtime import (
    rank_task_candidates,
    select_weighted_source_list,
    split_turn_into_task_shards,
)


def _result(
    *,
    content: str,
    final_score: float,
    importance: float = 0.5,
    metadata: dict | None = None,
) -> dict:
    return {
        "id": content.lower().replace(" ", "-"),
        "final_score": final_score,
        "memory": {
            "content": content,
            "importance": importance,
            "metadata": metadata or {},
        },
    }


def test_split_turn_into_task_shards_handles_multi_intent_turn():
    shards = split_turn_into_task_shards("아 오늘 뉴스 뭐가 있어? 아 맞다 그리고 경마 예상도")

    assert [shard.text for shard in shards] == ["오늘 뉴스 뭐가 있어?", "경마 예상도"]


def test_rank_task_candidates_prefers_seeded_source_list_for_analysis_start():
    persona_path = ["horseracing_specialist", "ai_maseonsang"]
    trigger_family = "analysis_start"
    results = [
        _result(
            content="generic style memory",
            final_score=0.92,
            metadata={"memory_kind": "style_hint", "importance_base": 0.02},
        ),
        _result(
            content="horseracing source list",
            final_score=0.73,
            metadata={
                "memory_kind": "source_list",
                "importance_base": 0.15,
                "trigger_weights": {"analysis_start": 0.6},
                "persona_trigger_weights": {"ai_maseonsang:analysis_start": 0.25},
            },
        ),
        _result(
            content="late-card lesson",
            final_score=0.79,
            metadata={
                "memory_kind": "lesson",
                "importance_base": 0.18,
                "trigger_weights": {"no_data_claim": 0.7},
            },
        ),
    ]

    ranked = rank_task_candidates(
        results,
        persona_path=persona_path,
        trigger_family=trigger_family,
        desired_kind="source_list",
    )

    assert ranked[0]["memory"]["content"] == "horseracing source list"
    assert ranked[0]["task_score"] > ranked[1]["task_score"]


def test_select_weighted_source_list_can_change_by_trigger_family():
    persona_path = ["horseracing_specialist", "ai_maseonsang"]
    source_memory = _result(
        content="seeded horseracing sources",
        final_score=0.81,
        metadata={
            "memory_kind": "source_list",
            "importance_base": 0.1,
            "trigger_weights": {"analysis_start": 0.55, "no_data_claim": 0.55},
            "source_rankings": {
                "analysis_start": [
                    {"source": "Gumvit chulma", "weight": 0.95, "role": "current_card"},
                    {"source": "KRA official schedule", "weight": 0.88, "role": "official"},
                    {"source": "Gumvit boards", "weight": 0.41, "role": "community_delta"},
                ],
                "no_data_claim": [
                    {"source": "KRA official schedule", "weight": 0.97, "role": "ground_zero"},
                    {"source": "Gumvit chulma", "weight": 0.90, "role": "current_card"},
                    {"source": "Gumvit boards", "weight": 0.35, "role": "community_delta"},
                ],
            },
        },
    )

    analysis_sources = select_weighted_source_list(
        [source_memory],
        persona_path=persona_path,
        trigger_family="analysis_start",
    )
    no_data_sources = select_weighted_source_list(
        [source_memory],
        persona_path=persona_path,
        trigger_family="no_data_claim",
    )

    assert analysis_sources[0]["source"] == "Gumvit chulma"
    assert no_data_sources[0]["source"] == "KRA official schedule"


def test_select_weighted_source_list_applies_persona_specific_override():
    results = [
        _result(
            content="seeded horseracing sources",
            final_score=0.8,
            metadata={
                "memory_kind": "source_list",
                "importance_base": 0.1,
                "trigger_weights": {"analysis_start": 0.4},
                "source_rankings": {
                    "analysis_start": [
                        {"source": "Gumvit chulma", "weight": 0.93, "role": "current_card"},
                        {"source": "KRA official schedule", "weight": 0.92, "role": "official"},
                    ]
                },
                "persona_source_overrides": {
                    "ai_masonyeo": {
                        "analysis_start": {
                            "kra official schedule": 0.08,
                        }
                    }
                },
            },
        )
    ]

    male_sources = select_weighted_source_list(
        results,
        persona_path=["horseracing_specialist", "ai_maseonsang"],
        trigger_family="analysis_start",
    )
    female_sources = select_weighted_source_list(
        results,
        persona_path=["horseracing_specialist", "ai_masonyeo"],
        trigger_family="analysis_start",
    )

    assert male_sources[0]["source"] == "Gumvit chulma"
    assert female_sources[0]["source"] == "KRA official schedule"


def test_multi_intent_turn_can_support_separate_task_specific_retrieval():
    shards = split_turn_into_task_shards("오늘 뉴스 뭐가 있어? 그리고 경마 예상도")
    assert [shard.text for shard in shards] == ["오늘 뉴스 뭐가 있어?", "경마 예상도"]

    news_results = [
        _result(
            content="news source list",
            final_score=0.79,
            metadata={
                "memory_kind": "source_list",
                "importance_base": 0.1,
                "trigger_weights": {"analysis_start": 0.55},
                "persona_trigger_weights": {"news_specialist:analysis_start": 0.2},
            },
        )
    ]
    horseracing_results = [
        _result(
            content="horseracing source list",
            final_score=0.79,
            metadata={
                "memory_kind": "source_list",
                "importance_base": 0.1,
                "trigger_weights": {"analysis_start": 0.55},
                "persona_trigger_weights": {"ai_maseonsang:analysis_start": 0.2},
            },
        )
    ]

    news_ranked = rank_task_candidates(
        news_results,
        persona_path=["news_specialist"],
        trigger_family="analysis_start",
        desired_kind="source_list",
    )
    racing_ranked = rank_task_candidates(
        horseracing_results,
        persona_path=["horseracing_specialist", "ai_maseonsang"],
        trigger_family="analysis_start",
        desired_kind="source_list",
    )

    assert news_ranked[0]["memory"]["content"] == "news source list"
    assert racing_ranked[0]["memory"]["content"] == "horseracing source list"
