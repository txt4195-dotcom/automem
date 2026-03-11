from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from automem.classification.memory_classifier import MemoryClassifier
from automem.config import normalize_memory_type


def build_classifier(*, client=None, model: str = "gpt-4o-mini") -> MemoryClassifier:
    logger = MagicMock()
    ensure_openai_client = MagicMock()
    get_openai_client = MagicMock(return_value=client)
    return MemoryClassifier(
        normalize_memory_type=normalize_memory_type,
        ensure_openai_client=ensure_openai_client,
        get_openai_client=get_openai_client,
        classification_model=model,
        logger=logger,
    )


def test_heuristic_classification_uses_patterns_without_llm():
    client = MagicMock()
    classifier = build_classifier(client=client)

    memory_type, confidence, _ = classifier.classify("I decided to use FalkorDB over ArangoDB")

    assert memory_type == "Decision"
    assert confidence >= 0.6
    client.chat.completions.create.assert_not_called()


def test_llm_fallback_classifies_when_no_pattern_matches():
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"type":"Insight","confidence":0.82}')
            )
        ]
    )
    classifier = build_classifier(client=client)

    memory_type, confidence, _ = classifier.classify("Late source arrival changed the whole read.")

    assert memory_type == "Insight"
    assert confidence == 0.82
    client.chat.completions.create.assert_called_once()
    call = client.chat.completions.create.call_args
    assert call.kwargs["response_format"] == {"type": "json_object"}
    assert "Classify each memory into exactly ONE" in call.kwargs["messages"][0]["content"]


def test_llm_alias_is_normalized_to_canonical_type():
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"type":"analysis","confidence":0.74}')
            )
        ]
    )
    classifier = build_classifier(client=client)

    memory_type, confidence, _ = classifier.classify("This became a durable lesson about delays.")

    assert memory_type == "Insight"
    assert confidence == 0.74


def test_invalid_llm_response_falls_back_to_memory():
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="not-json"))]
    )
    classifier = build_classifier(client=client)

    memory_type, confidence, _ = classifier.classify("A late card can flip the whole day.")

    assert memory_type == "Memory"
    assert confidence == 0.3
