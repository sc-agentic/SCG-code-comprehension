import json
import os
from typing import Dict, Any
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class MetricsLogger:
    def __init__(self, log_file: str = "metrics_log.json"):
        self.log_file = log_file
        self.embedding_model = None
        self._init_embedding_model()

    def _init_embedding_model(self):
        try:
            if self.embedding_model is None:
                print("Loading embedding model")
                self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                print("Embedding model loaded successfully")
        except Exception as e:
            print(f"Could not load embedding model: {e}")
            import traceback
            traceback.print_exc()

    def classify_question_type(self, question: str) -> str:
        question_lower = question.lower()
        if any(word in question_lower for word in ['what is', 'describe', 'explain', 'definition', 'purpose']):
            return "definition"
        elif any(word in question_lower for word in ['where', 'used', 'usage', 'called', 'invoked', 'references']):
            return "usage"
        elif any(word in question_lower for word in ['how does', 'implementation', 'algorithm', 'logic', 'works']):
            return "implementation"
        elif any(word in question_lower for word in ['test', 'testing', 'junit', 'mock', 'verify']):
            return "testing"
        elif any(word in question_lower for word in ['error', 'exception', 'throw', 'catch', 'handling']):
            return "exception"
        else:
            return "general"

    def calculate_answer_relevancy(self, question: str, answer: str) -> float:
        if not self.embedding_model or not answer.strip():
            return 0.0
        try:
            q_emb = self.embedding_model.encode([question])
            a_emb = self.embedding_model.encode([answer])
            similarity = cosine_similarity(q_emb, a_emb)[0][0]
            return max(0.0, min(1.0, similarity))
        except Exception:
            return 0.0

    def calculate_faithfulness(self, answer: str, context: str) -> float:
        if not self.embedding_model or not answer.strip() or not context.strip():
            return 0.0
        try:
            answer_sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 10]
            if not answer_sentences:
                return 0.0
            faithful_count = 0
            for sentence in answer_sentences:
                sentence_emb = self.embedding_model.encode([sentence])
                context_emb = self.embedding_model.encode([context])
                similarity = cosine_similarity(sentence_emb, context_emb)[0][0]
                if similarity > 0.3:
                    faithful_count += 1
            return faithful_count / len(answer_sentences)
        except Exception:
            return 0.0

    def calculate_context_precision(self, question: str, context: str) -> float:
        if not self.embedding_model or not context.strip():
            return 0.0
        try:
            sections = [s.strip() for s in context.split('##') if s.strip()]
            if len(sections) <= 1:
                return 1.0
            question_emb = self.embedding_model.encode([question])
            similarities = []
            for section in sections:
                if len(section) > 20:
                    section_emb = self.embedding_model.encode([section])
                    sim = cosine_similarity(question_emb, section_emb)[0][0]
                    similarities.append(sim)
            if len(similarities) <= 1:
                return 1.0
            correctly_ordered = 0
            for i in range(len(similarities) - 1):
                if similarities[i] >= similarities[i + 1]:
                    correctly_ordered += 1
            return correctly_ordered / (len(similarities) - 1)
        except Exception:
            return 0.0

    def calculate_context_recall(self, question: str, context: str) -> float:
        if not context.strip():
            return 0.0
        question_lower = question.lower()
        context_lower = context.lower()
        expected_elements = []
        if any(word in question_lower for word in ['usage', 'used', 'where', 'called']):
            expected_elements = ['controller', 'service', 'method', 'endpoint', 'call']
        elif any(word in question_lower for word in ['test', 'testing']):
            expected_elements = ['test', '@test', 'junit', 'mock', 'assert']
        elif any(word in question_lower for word in ['exception', 'error', 'handling']):
            expected_elements = ['exception', 'throw', 'catch', 'error', 'try']
        elif any(word in question_lower for word in ['what', 'describe', 'definition']):
            expected_elements = ['class', 'method', 'public', 'private', 'interface']
        elif any(word in question_lower for word in ['how', 'implementation', 'algorithm']):
            expected_elements = ['implementation', 'algorithm', 'logic', 'process']
        else:
            return 0.7
        found_elements = sum(1 for elem in expected_elements if elem in context_lower)
        return found_elements / len(expected_elements) if expected_elements else 0.5

    def calculate_context_relevancy(self, question: str, context: str) -> float:
        if not self.embedding_model or not context.strip():
            return 0.0
        try:
            sections = [s.strip() for s in context.split('##') if s.strip()]
            if not sections:
                sections = [context]
            question_emb = self.embedding_model.encode([question])
            relevant_sections = 0
            for section in sections:
                if len(section) > 20:
                    section_emb = self.embedding_model.encode([section])
                    similarity = cosine_similarity(question_emb, section_emb)[0][0]
                    if similarity > 0.25:
                        relevant_sections += 1
            total_sections = len([s for s in sections if len(s) > 20])
            return relevant_sections / total_sections if total_sections > 0 else 0.0
        except Exception:
            return 0.0

    def evaluate_response(self, question: str, answer: str, context: str,
                          processing_time: float, model_name: str = None, additional_data: Dict = None) -> Dict[str, Any]:

        question_type = self.classify_question_type(question)
        metrics = {}
        if self.embedding_model:
            try:
                metrics = {
                    "answer_relevancy": float(self.calculate_answer_relevancy(question, answer)),
                    "faithfulness": float(self.calculate_faithfulness(answer, context)),
                    "context_precision": float(self.calculate_context_precision(question, context)),
                    "context_recall": float(self.calculate_context_recall(question, context)),
                    "context_relevancy": float(self.calculate_context_relevancy(question, context))
                }
                metrics["overall_score"] = float(sum(metrics.values()) / len(metrics))
            except Exception as e:
                print(f"Error calculating metrics: {e}")

        else:
            print("Embedding model not available, skipping metrics calculation")
        evaluation_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "question_type": question_type,
            "answer": answer,
            "context_length": len(context),
            "processing_time": processing_time,
            "model_name": model_name or "unknown",
            "metrics": metrics,
            "metadata": additional_data or {}
        }
        return evaluation_entry

    def log_metrics(self, question: str, answer: str, context: str, processing_time: float, model_name: str = None, additional_data: Dict = None):

        evaluation_entry = self.evaluate_response(question, answer, context, processing_time, model_name, additional_data)
        evaluations = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    evaluations = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                evaluations = []
        evaluations.append(evaluation_entry)
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(evaluations, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving evaluation log: {e}")


metrics_logger = MetricsLogger("metrics_log.json")
