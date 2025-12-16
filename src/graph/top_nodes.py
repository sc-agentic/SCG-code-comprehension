import json
import re
import time
from typing import Any, Dict

from loguru import logger

from context.context_builder import build_context
from core.models import IntentAnalysis
from graph.NeighborTypeEnum import NeighborTypeEnum
from graph.QueryTopMode import QueryTopMode
from src.clients.llm_client import call_llm


def get_metric_value(node: Dict[str, Any], metric: str) -> float:
    """
    Returns a numeric metric value for a node.

    Args:
        node: Node metadata dictionary
        metric: Metric key (e.g., "combined", "pagerank", "in-degree", "out-degree",
            "number_of_neighbors")

    Returns:
        Metric value for the node
    """
    if metric == "number_of_neighbors":
        related_entities_str = node.get("related_entities", "")
        try:
            related_entities = (
                json.loads(related_entities_str)
                if isinstance(related_entities_str, str)
                else related_entities_str
            )
        except (json.JSONDecodeError, TypeError):
            logger.debug("Failed to parse related_entities, using empty list.")
            related_entities = []
        return len(related_entities)
    else:
        return float(node.get(metric, 0.0))


def extract_json(text: str) -> dict:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found")
    return json.loads(match.group())

async def get_top_nodes_context(
    question: str, analysis: IntentAnalysis, model_name: str, collection: Any, **params
) -> tuple[list[dict[str, float | Any]], str]:
    """
    Finds top nodes based on LLM-guided kind/metric selection.

    Args:
        question: Natural-language question
        collection: Chroma collection handle

    Returns:
        Top nodes with metadata and metric values
    """
    query_mode = params.get("query_mode", QueryTopMode.LIST_ONLY)
    kinds = params.get("kinds", "ANY")
    if isinstance(kinds, str):
        kinds = [kinds]
    kinds = [NeighborTypeEnum[kind.upper()] for kind in kinds]
    metric = params.get("metric", "combined")

    if isinstance(query_mode, str):
        try:
            query_mode = QueryTopMode(query_mode)
        except ValueError:
            logger.warning(f"Invalid query_mode '{query_mode}', using LIST_ONLY.")
            query_mode = QueryTopMode.LIST_ONLY

    classification_prompt = f"""
    User question: "{question}"

    Your task:
    Analyze user question following rules specified below (bear in mind that question can be in polish):
    
    1. Determine number of nodes ("limit"):
        - "all", "everything", "wszystkie" is in question -> "limit" = "all"
        - If the question contains a number connected to number of nodes -> "limit" = that number
        
    2. Exact metric value ("exact_metric_value"):
        - If limit = "all" AND user explicitly mentions a metric value (example: "with none neighbors", "with 0 neighbors", "with 0 lines of code") → metric_value = that value
            -Treat words like "none", "no", "without", "brak" as 0
        - Otherwise → metric_value = 0
        
    3. Sort order ("order"):
       - If question contains words like "biggest", "largest", "most", "max" → use "desc"
       - If question contains words like "smallest", "least", "min" → use "asc"
       - If not sure → order = "desc"
    
    Rules:
    - ALWAYS return valid JSON only
    - DO NOT include comments, explanations, or extra text
    - Do NOT infer or guess anything outside the rules above

    Return ONLY valid JSON format:
    {{"limit": 5, "order": "desc", "exact_metric_value": 0}}
    """
    start_time = time.time()
    analysis = await call_llm(classification_prompt)
    logger.debug(f"Top nodes analysis: {analysis}")
    try:
        parsed = extract_json(analysis)
        order = parsed.get("order", "desc")
        limit = parsed.get("limit", 5)
        exact_metric_value = parsed.get("exact_metric_value", 0)
    except json.JSONDecodeError:
        order = "desc"
        limit = 5
        exact_metric_value = 0

    logger.info(f"Parsed: {parsed}")

    results = collection.get(include=["metadatas", "documents"])

    nodes = [
        {
            "node": results["ids"][i],
            "metadata": results["metadatas"][i],
            "code": results["documents"][i],
            "metric_value": get_metric_value(results["metadatas"][i], metric),
        }
        for i in range(len(results["ids"]))
    ]
    if limit == 'all':
        top_nodes = [node for node in nodes if (NeighborTypeEnum[node["metadata"].get(
            "kind").upper()] in kinds or NeighborTypeEnum.ANY in kinds) and node["metric_value"] == exact_metric_value]
    else:
        top_nodes = sorted(
            (node for node in nodes if
             NeighborTypeEnum[node["metadata"].get("kind").upper()] in kinds or NeighborTypeEnum.ANY in kinds),
            key=lambda n: n["metric_value"],
            reverse=(order.lower() == "desc"),
        )[:limit]

    logger.info(f"Top nodes: {top_nodes}")

    logger.debug(f"MODE: {query_mode}")
    if query_mode == QueryTopMode.LIST_ONLY:
        context = " ".join(
            f"{node.get('metadata', {}).get('label', '')} - {node.get('metric_value'):.2f}"
            for node in top_nodes
        )
    else:
        top_nodes = [
            (
                n["metric_value"],
                {
                    "node": n["node"],
                    "metadata": n["metadata"],
                    "code": n["code"],
                },
            )
            for n in top_nodes
        ]
        context = build_context(top_nodes, "definition", 1.0, question=question, target_method=None)

    logger.debug(f"Top query context: {context}")
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000
    logger.debug(f"Completed in: {elapsed_ms:.1f}ms")
    return top_nodes, context or "<NO CONTEXT FOUND>"
