import json
import time
from typing import Any, Dict, List, Tuple, Optional
from loguru import logger
from core.models import IntentAnalysis, IntentCategory
from graph.NeighborTypeEnum import NeighborTypeEnum
from graph.generate_embeddings_graph import generate_embeddings_graph
from graph.retrieval_utils import get_embedding_inputs, query_embeddings, expand_definition_neighbors
from graph.model_cache import get_graph_model
from src.graph.retriver import extract_key_value_pairs_simple


async def get_related_entities(
        question: str,
        analysis: IntentAnalysis,
        model_name: str,
        collection: Any,
        **params,
) -> Tuple[List[Tuple[float, Dict[str, Any]]], str]:
    """
    Find nodes included in user question

    Args:
        question: Natural-language question
        analysis: Analysis of user's question
        model_name: Name of LLM model used for finding context
        collection: Chroma collection handle
        **params:
            top_k: Maximum number of top nodes to get context from
            max_neighbors: Maximum number of neighbors for each top node

    Returns:
        Top nodes with metadata and metric values
    """
    limit = params.get("limit", 5)
    neighbor_types = params.get("neighbor_types", "ANY")
    if isinstance(neighbor_types, str):
        neighbor_types = [neighbor_types]
    neighbor_types = [NeighborTypeEnum[type.upper()] for type in neighbor_types]
    logger.info(f"Limit: {limit}, Neighbor types: {neighbor_types}")

    pairs = await extract_key_value_pairs_simple(question)
    pairs = [(t.lower(), n.lower()) for t, n in pairs]

    logger.debug(f"Question: '{question}'")
    logger.debug(f"Extracted pairs: {pairs}")

    embeddings_input = get_embedding_inputs(pairs, question)
    logger.debug(f"Embedding input: {embeddings_input}")

    get_graph_model()

    query_embeddings_var = generate_embeddings_graph(embeddings_input, model_name)
    results = query_embeddings(collection, query_embeddings_var, embeddings_input, pairs)

    context = ""

    neighbor_type_filter = [
        nt.value.upper()
        for nt in neighbor_types
        if nt != NeighborTypeEnum.ANY
    ]

    for result in results:
        node_id = result[1]["node"]
        metadata = result[1]["metadata"]
        related_entities_str = metadata.get("related_entities", "")
        try:
            related_entities = (
                json.loads(related_entities_str)
                if isinstance(related_entities_str, str)
                else related_entities_str
            )
        except (json.JSONDecodeError, TypeError):
            related_entities = []

        if not related_entities:
            continue

        logger.debug(f"related_entities: {related_entities}")
        if NeighborTypeEnum.ANY in neighbor_types:
            neighbors = collection.get(
                ids=related_entities,
                include=["metadatas", "documents"],
            )
        else:
            neighbors = collection.get(
                ids=related_entities,
                include=["metadatas", "documents"],
                where={"kind": {"$in": neighbor_type_filter}},
            )

        combined_neighbors = list(zip(neighbors["ids"], neighbors["metadatas"], neighbors["documents"]))

        neighbors_sorted = sorted(
            combined_neighbors,
            key=lambda x: x[1].get("combined", 0),
            reverse=True
        )

        if limit == "all":
            neighbors_sorted = neighbors_sorted
        else:
            neighbors_sorted = neighbors_sorted[:limit]

        logger.info(f"Neighbors: {len(neighbors_sorted)}")
        context += f"Neighbors (sorted by combined metric importance) of Node: {node_id}, with name: {metadata['node']} kind: {metadata['kind']} located in: {metadata['uri']}:\n"

        for i, (node_id, neighbors_metadata, _) in enumerate(neighbors_sorted):
            context += f"{i + 1}.Node_id: {node_id}, name: {neighbors_metadata.get('label', '')}, kind: {neighbors_metadata.get('kind', '')}, uri: {neighbors_metadata.get('uri', '')}\n"

    logger.info(f"Context: {context}")
    return [], context or "<NO CONTEXT FOUND>"
