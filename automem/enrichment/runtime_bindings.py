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
from automem.enrichment.runtime_helpers import (
    score_semantic_edges as _score_semantic_edges_runtime,
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

    def score_semantic_edges(
        graph: Any,
        memory_id: str,
        content: str,
        semantic_neighbors: List[Tuple[str, float]],
    ) -> int:
        return _score_semantic_edges_runtime(
            graph=graph,
            memory_id=memory_id,
            source_content=content,
            semantic_neighbors=semantic_neighbors,
            get_openai_client_fn=get_openai_client_fn,
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

    def observer_reverse_inference(
        graph: Any,
        memory_id: str,
        semantic_neighbors: List[Tuple[str, float]],
        tags: List[str],
        importance: float = 0.5,
    ) -> None:
        """Backpropagate observer values from what the user chose to store.

        importance = significance (error magnitude). High importance = strong
        gradient, low importance = weak gradient. The mismatch between
        nano's edge scores (target) and observer values (current estimate)
        is the error signal, scaled by importance.
        """
        observer_id = None
        for tag in tags:
            if tag.startswith("user:") or tag.startswith("agent:"):
                observer_id = tag
                break
        if not observer_id:
            return

        from automem.enrichment.edge_scoring import SCORE_EDGE_TYPES
        from automem.observer.recall import fetch_score_edges
        from automem.observer.vector import (
            get_or_create_observer,
            infer_observer_from_store,
            save_observer,
        )

        timestamp = utc_now_fn()
        observer = get_or_create_observer(
            graph=graph, observer_id=observer_id, timestamp=timestamp,
        )

        all_layers = list(SCORE_EDGE_TYPES.keys())
        edge_scores_list = []
        for neighbor_id, _score in semantic_neighbors:
            scores = fetch_score_edges(
                graph=graph,
                source_id=memory_id,
                target_id=neighbor_id,
                layers=all_layers,
            )
            if scores:
                edge_scores_list.append(scores)

        if edge_scores_list:
            deltas = infer_observer_from_store(
                observer=observer,
                edge_scores_list=edge_scores_list,
                significance=importance,
            )
            save_observer(graph=graph, observer=observer, timestamp=timestamp)
            logger.debug(
                "Observer %s updated via reverse inference from %s "
                "(significance=%.2f, %d edge sets, %d dims moved)",
                observer_id, memory_id[:8], importance,
                len(edge_scores_list),
                sum(len(d) for d in deltas.values()),
            )

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
            score_semantic_edges_fn=score_semantic_edges,
            score_node_fn=score_node,
            observer_reverse_inference_fn=observer_reverse_inference,
            enrichment_enable_summaries=enrichment_enable_summaries,
            generate_summary_fn=generate_summary_fn,
            utc_now_fn=utc_now_fn,
            collection_name=collection_name,
            unexpected_response_exc=unexpected_response_exc,
            logger=logger,
        )

    return EnrichmentRuntimeBindings(
        jit_enrich_lightweight=jit_enrich_lightweight,
        enrich_memory=enrich_memory,
        temporal_cutoff=temporal_cutoff,
        find_temporal_relationships=find_temporal_relationships,
        detect_patterns=detect_patterns,
        link_semantic_neighbors=link_semantic_neighbors,
    )
