import json
import os
import re
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer
from chroma_client import get_chroma_client, get_collection, default_collection_name
from graph.generate_embeddings_graph import generate_embeddings_graph
from intent_analyzer import get_intent_analyzer, classify_question

default_classifier_embeddings_path = "embeddings/classifier_example_embeddings.json"
default_classifier_model = "sentence-transformers/all-MiniLM-L6-v2"
default_chroma_path = "embeddings/chroma_storage"
default_codebert_model = "microsoft/codebert-base"
default_top_k = 7


def load_classifier_embeddings(path: str = None) -> dict:
    embeddings_path = path or default_classifier_embeddings_path
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Classifier embeddings not found")
    with open(embeddings_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_classifier_model(model_name: str = None) -> SentenceTransformer:
    model_name = model_name or default_classifier_model
    return SentenceTransformer(model_name)


try:
    classifier_embeddings = load_classifier_embeddings()
    classifier_model = get_classifier_model()
except Exception as e:
    print(f"Error with loading classifier components: {e}")
    classifier_embeddings = {}
    classifier_model = None

chroma_client = get_chroma_client(storage_path=default_chroma_path)


def extract_key_value_pairs_simple(question: str) -> List[Tuple[str, str]]:
    pairs = []
    question_lower = question.lower()
    words = question_lower.split()
    key_terms = {"class", "method", "function", "variable", "property"}
    for i, word in enumerate(words):
        if word in key_terms:
            if i + 1 < len(words):
                next_word = words[i + 1]
                clean_name = next_word.strip("'\"().,!?")
                if clean_name and len(clean_name) > 1:
                    pairs.append((word, clean_name))
            elif i > 0:
                prev_word = words[i - 1]
                clean_name = prev_word.strip("'\"().,!?")
                if clean_name and len(clean_name) > 1:
                    pairs.append((word, clean_name))
    java_class_pattern = r'\b(\w+(?:service|controller|repository|dto|entity|exception))\b'
    java_matches = re.findall(java_class_pattern, question_lower)
    for match in java_matches:
        pairs.append(('class', match))
    for word in re.findall(r'\b[A-Z][a-zA-Z]+\b', question):
        word_lower = word.lower()
        if (word_lower.endswith(('service', 'controller', 'repository', 'dto', 'entity', 'exception')) or
                len(word) > 8):
            pairs.append(('class', word_lower))
    method_with_parens = re.findall(r'\b(\w+)\s*\(\s*\)', question_lower)
    for method_name in method_with_parens:
        pairs.append(('method', method_name))
    method_prefixes = ['find', 'get', 'set', 'create', 'update', 'delete', 'add', 'remove']
    for word in words:
        clean_word = word.strip("'\"().,!?")
        for prefix in method_prefixes:
            if clean_word.startswith(prefix) and len(clean_word) > len(prefix):
                pairs.append(('method', clean_word))
                break
    pattern = r'(\w+)\s+method\s+in\s+(\w+)\s+class'
    matches = re.findall(pattern, question_lower)
    for method, class_name in matches:
        pairs.append(('method', method))
        pairs.append(('class', class_name))
    seen = set()
    unique_pairs = []
    for pair in pairs:
        if pair not in seen:
            unique_pairs.append(pair)
            seen.add(pair)
    return unique_pairs


def preprocess_question(q: str) -> str:
    q = re.sub(r'\bmethod\s+\w+\b', 'method', q, flags=re.IGNORECASE)
    q = re.sub(r'\bfunction\s+\w+\b', 'function', q, flags=re.IGNORECASE)
    q = re.sub(r'\bclass\s+\w+\b', 'class', q, flags=re.IGNORECASE)
    q = re.sub(r'\bvariable\s+\w+\b', 'variable', q, flags=re.IGNORECASE)
    q = re.sub(r'\s+', ' ', q).strip()
    return q.lower()


def similar_node(question: str, model_name: str = default_codebert_model, collection_name: str = default_collection_name, top_k: int = default_top_k) -> Tuple[List[Tuple[float, Dict[str, Any]]], str]:
    collection = get_collection("scg_embeddings")
    pairs = extract_key_value_pairs_simple(question)
    embeddings_input = []
    for key, value in pairs:
        embeddings_input.append(f"{key} {value}" if key else value)
    if not embeddings_input:
        embeddings_input = [question]
    query_embeddings = generate_embeddings_graph(embeddings_input, model_name)
    results = []
    for query_emb in query_embeddings:
        try:
            query_result = collection.query(
                query_embeddings=[query_emb.tolist()],
                n_results=top_k,
                include=["embeddings", "metadatas", "documents", "distances"])
            for i in range(len(query_result["ids"][0])):
                score = 1 - query_result["distances"][0][i]
                node_id = query_result["ids"][0][i]
                metadata = query_result["metadatas"][0][i]
                code = query_result["documents"][0][i]
                results.append((score, {
                    "node": node_id,
                    "metadata": metadata,
                    "code": code
                }))
        except Exception as e:
            print(f"Error querying collection: {e}")

    seen = set()
    unique_results = []
    for score, node in sorted(results, key=lambda x: -x[0]):
        if node["node"] not in seen:
            unique_results.append((score, node))
            seen.add(node["node"])
        if len(unique_results) >= len(embeddings_input) * top_k:
            break
    top_nodes = unique_results[:len(embeddings_input)]
    top_k_codes = [node["code"] for _, node in top_nodes if node["code"]]
    try:
        analyzer = get_intent_analyzer()
        analysis = analyzer.enhanced_classify_question(question)
        category = analysis.primary_intent.value
    except Exception as e:
        print(f"Fallback to basic classification: {e}")
        category = classify_question(preprocess_question(question))

    max_neighbors = {"general": 5, "medium": 3, "specific": 1}.get(category, 2)
    print(f"Category: {category}")

    all_neighbors_ids = set()
    for _, node in top_nodes:
        neighbors = node["metadata"].get("related_entities", [])
        if isinstance(neighbors, str):
            try:
                neighbors = json.loads(neighbors)
            except json.JSONDecodeError:
                neighbors = []
        all_neighbors_ids.update(neighbors)

    neighbor_codes = []
    if all_neighbors_ids:
        try:
            neighbor_nodes = collection.get(
                ids=list(all_neighbors_ids),
                include=["documents", "metadatas"]
            )

            neighbors_with_scores = []
            for i in range(len(neighbor_nodes["ids"])):
                nid = neighbor_nodes["ids"][i]
                meta = neighbor_nodes["metadatas"][i]
                doc = neighbor_nodes["documents"][i]

                if doc:
                    score = meta.get("combined", 0.0)
                    neighbors_with_scores.append((score, nid, doc))

            sorted_neighbors = sorted(neighbors_with_scores, key=lambda x: -x[0])
            neighbor_codes = [doc for _, _, doc in sorted_neighbors[:max_neighbors]]
        except Exception as e:
            print(f"Error getting neighbors: {e}")

    all_codes = []
    seen_codes = set()
    for code in top_k_codes + neighbor_codes:
        if code and code not in seen_codes and not code.startswith("<"):
            all_codes.append(code)
            seen_codes.add(code)

    full_context = "\n\n".join(all_codes)

    if not all_codes and category == "general":
        try:
            all_nodes = collection.get(include=["documents", "metadatas", "ids"])
            importance_scores = []
            for i in range(len(all_nodes["ids"])):
                doc = all_nodes["documents"][i]
                meta = all_nodes["metadatas"][i]
                nid = all_nodes["ids"][i]
                score = meta.get("importance", {}).get("combined", 0.0) if isinstance(meta.get("importance"), dict) else meta.get("combined", 0.0)
                if doc:
                    importance_scores.append((score, nid, doc))
            sorted_by_importance = sorted(importance_scores, key=lambda x: -x[0])
            fallback_docs = [doc for _, _, doc in sorted_by_importance[:5]]
            full_context = "\n\n".join(fallback_docs)
        except Exception as e:
            print(f"Error retrieving fallback for general question: {e}")
            full_context = "<NO CONTEXT FOUND>"
    return top_nodes, full_context or "<NO CONTEXT FOUND>"
