from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from automem.enrichment.runtime_helpers import detect_patterns as _detect_patterns_runtime
from automem.enrichment.runtime_helpers import (
    find_temporal_relationships as _find_temporal_relationships_runtime,
)
from automem.enrichment.runtime_helpers import (
    link_semantic_neighbors as _link_semantic_neighbors_runtime,
)
from automem.enrichment.node_scoring import (
    score_nodes_with_llm as _score_nodes_with_llm,
    save_node_scores as _save_node_scores,
)
from automem.enrichment.runtime_helpers import temporal_cutoff as _temporal_cutoff_runtime
from automem.enrichment.runtime_orchestration import enrich_memory as _enrich_memory_runtime
from automem.enrichment.runtime_orchestration import (
    jit_enrich_lightweight as _jit_enrich_lightweight_runtime,
)


@dataclass(frozen=True)
class EnrichmentRuntimeBindings:
    jit_enrich_lightweight: Callable[[str, Dict[str, Any]], Optional[Dict[str, Any]]]
    enrich_memory: Callable[..., bool]
    temporal_cutoff: Callable[[], str]
    find_temporal_relationships: Callable[[Any, str, int], int]
    detect_patterns: Callable[[Any, str, str], List[Dict[str, Any]]]
    link_semantic_neighbors: Callable[[Any, str], List[Tuple[str, float]]]


def create_enrichment_runtime(
    *,
    get_memory_graph_fn: Callable[[], Any],
    get_qdrant_client_fn: Callable[[], Any],
    get_openai_client_fn: Callable[[], Any],
    parse_metadata_field_fn: Callable[[Any], Any],
    normalize_tag_list_fn: Callable[[Any], List[str]],
    extract_entities_fn: Callable[[str], Dict[str, List[str]]],
    slugify_fn: Callable[[str], str],
    compute_tag_prefixes_fn: Callable[[List[str]], List[str]],
    classify_memory_fn: Callable[[str], Tuple[str, float, Any]],
    search_stopwords: Set[str],
    enrichment_enable_summaries: bool,
    generate_summary_fn: Callable[[str, Any], Any],
    utc_now_fn: Callable[[], str],
    collection_name: str,
    enrichment_similarity_limit: int,
    enrichment_similarity_threshold: float,
    unexpected_response_exc: Any,
    logger: Any,
) -> EnrichmentRuntimeBindings:
    def temporal_cutoff() -> str:
        return _temporal_cutoff_runtime()

    def find_temporal_relationships(graph: Any, memory_id: str, limit: int = 5) -> int:
        return _find_temporal_relationships_runtime(
            graph=graph,
            memory_id=memory_id,
            limit=limit,
            cutoff_fn=temporal_cutoff,
            utc_now_fn=utc_now_fn,
            logger=logger,
        )

    def detect_patterns(graph: Any, memory_id: str, content: str) -> List[Dict[str, Any]]:
        return _detect_patterns_runtime(
            graph=graph,
            memory_id=memory_id,
            content=content,
            classify_fn=classify_memory_fn,
            search_stopwords=search_stopwords,
            utc_now_fn=utc_now_fn,
            logger=logger,
        )

    def link_semantic_neighbors(graph: Any, memory_id: str) -> List[Tuple[str, float]]:
        return _link_semantic_neighbors_runtime(
            graph=graph,
            memory_id=memory_id,
            get_qdrant_client_fn=get_qdrant_client_fn,
            collection_name=collection_name,
            similarity_limit=enrichment_similarity_limit,
            similarity_threshold=enrichment_similarity_threshold,
            utc_now_fn=utc_now_fn,
            logger=logger,
        )

    def jit_enrich_lightweight(
        memory_id: str, properties: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        return _jit_enrich_lightweight_runtime(
            memory_id=memory_id,
            properties=properties,
            get_memory_graph_fn=get_memory_graph_fn,
            get_qdrant_client_fn=get_qdrant_client_fn,
            parse_metadata_field_fn=parse_metadata_field_fn,
            normalize_tag_list_fn=normalize_tag_list_fn,
            extract_entities_fn=extract_entities_fn,
            slugify_fn=slugify_fn,
            compute_tag_prefixes_fn=compute_tag_prefixes_fn,
            enrichment_enable_summaries=enrichment_enable_summaries,
            generate_summary_fn=generate_summary_fn,
            utc_now_fn=utc_now_fn,
            collection_name=collection_name,
            logger=logger,
        )

    def score_node(graph: Any, memory_id: str, content: str) -> bool:
        """Score a single node across 7 type-weight dimensions using nano."""
        openai_client = get_openai_client_fn()
        if openai_client is None:
            return False
        results = _score_nodes_with_llm(
            openai_client=openai_client,
            memories=[{"id": memory_id, "content": content}],
        )
        if results:
            return _save_node_scores(graph, results[0])
        return False

    def _generate_reasoning(content: str):
        from automem.utils.reasoning_generator import generate_reasoning
        return generate_reasoning(content)

    def _re_embed(memory_id: str, rich_text: str):
        """Re-embed a memory with enriched text (content + words).

        Uses update_vectors to replace only the vector, preserving existing payload.
        """
        from automem.embedding import get_provider
        provider = get_provider()
        embedding = provider.embed(rich_text)
        qdrant = get_qdrant_client_fn()
        if qdrant is None:
            return
        from qdrant_client.models import PointVectors
        qdrant.update_vectors(
            collection_name=collection_name,
            points=[PointVectors(id=memory_id, vector=embedding)],
        )

    def _generate_scenarios(content, enrichment_json):
        from automem.utils.scenario_generator import generate_scenarios
        return generate_scenarios(content, enrichment_json)

    def _store_scenario(parent_id, scenario, parent_tags):
        """Store a scenario as a separate Memory node linked to parent."""
        import uuid
        from automem.utils.scenario_generator import build_scenario_content

        graph = get_memory_graph_fn()
        if graph is None:
            return None

        scenario_id = str(uuid.uuid4())
        scenario_content = build_scenario_content(scenario)
        scenario_tags = list(parent_tags) + ["kind:scenario"]

        # Store in FalkorDB
        graph.query(
            """
            CREATE (m:Memory {
                id: $id,
                content: $content,
                type: 'Scenario',
                tags: $tags,
                importance: 0.5,
                timestamp: $ts,
                processed: true,
                enriched: true
            })
            """,
            {
                "id": scenario_id,
                "content": scenario_content,
                "tags": scenario_tags,
                "ts": utc_now_fn(),
            },
        )

        # Link to parent
        graph.query(
            """
            MATCH (child:Memory {id: $child_id}), (parent:Memory {id: $parent_id})
            MERGE (child)-[:DERIVED_FROM]->(parent)
            """,
            {"child_id": scenario_id, "parent_id": parent_id},
        )

        # Store in Qdrant
        try:
            from automem.embedding import get_provider
            provider = get_provider()
            embedding = provider.embed(scenario_content)
            qdrant = get_qdrant_client_fn()
            if qdrant is not None:
                from qdrant_client.models import PointStruct
                qdrant.upsert(
                    collection_name=collection_name,
                    points=[PointStruct(
                        id=scenario_id,
                        vector=embedding,
                        payload={
                            "tags": scenario_tags,
                            "metadata": {"derived_from": parent_id, "scenario": scenario},
                        },
                    )],
                )
        except Exception:
            logger.debug("Qdrant upsert for scenario %s failed (non-fatal)", scenario_id[:8])

        return scenario_id

    def enrich_memory(memory_id: str, *, forced: bool = False) -> bool:
        return _enrich_memory_runtime(
            memory_id=memory_id,
            forced=forced,
            get_memory_graph_fn=get_memory_graph_fn,
            get_qdrant_client_fn=get_qdrant_client_fn,
            parse_metadata_field_fn=parse_metadata_field_fn,
            normalize_tag_list_fn=normalize_tag_list_fn,
            extract_entities_fn=extract_entities_fn,
            slugify_fn=slugify_fn,
            compute_tag_prefixes_fn=compute_tag_prefixes_fn,
            find_temporal_relationships_fn=find_temporal_relationships,
            detect_patterns_fn=detect_patterns,
            link_semantic_neighbors_fn=link_semantic_neighbors,
            score_node_fn=score_node,
            enrichment_enable_summaries=enrichment_enable_summaries,
            generate_summary_fn=generate_summary_fn,
            utc_now_fn=utc_now_fn,
            collection_name=collection_name,
            unexpected_response_exc=unexpected_response_exc,
            logger=logger,
            generate_reasoning_fn=_generate_reasoning,
            re_embed_fn=_re_embed,
            generate_scenarios_fn=_generate_scenarios,
            store_scenario_fn=_store_scenario,
        )

    return EnrichmentRuntimeBindings(
        jit_enrich_lightweight=jit_enrich_lightweight,
        enrich_memory=enrich_memory,
        temporal_cutoff=temporal_cutoff,
        find_temporal_relationships=find_temporal_relationships,
        detect_patterns=detect_patterns,
        link_semantic_neighbors=link_semantic_neighbors,
    )
