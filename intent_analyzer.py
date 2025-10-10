import json
import re
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from models import IntentCategory, ExpertiseLevel, IntentAnalysis


class IntentAnalyzer:
    def __init__(self, classifier_embeddings_path: str = "embeddings/classifier_example_embeddings.json", classifier_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.classifier_embeddings_path = classifier_embeddings_path
        self.classifier_model_name = classifier_model
        self._classifier_model: Optional[SentenceTransformer] = None
        self._classifier_embeddings: Optional[Dict] = None

        self.intent_patterns = {
            IntentCategory.USAGE: {
                "keywords": ["used", "called", "invoked", "where", "references", "usage", "utilized"],
                "patterns": [r'where.*used', r'how.*called', r'.*usage.*', r'.*referenced.*'],
                "weight": 2.0
            },
            IntentCategory.DEFINITION: {
                "keywords": ["what is", "describe", "explain", "define", "meaning", "purpose"],
                "patterns": [r'what\s+is', r'describe.*', r'explain.*', r'define.*'],
                "weight": 1.8
            },
            IntentCategory.IMPLEMENTATION: {
                "keywords": ["how does", "implementation", "algorithm", "logic", "works", "internally"],
                "patterns": [r'how\s+does.*work', r'implementation.*', r'algorithm.*', r'.*logic.*'],
                "weight": 1.6
            },
            IntentCategory.TESTING: {
                "keywords": ["test", "testing", "junit", "mock", "verify", "unit test"],
                "patterns": [r'.*test.*', r'junit.*', r'mock.*', r'.*testing.*'],
                "weight": 1.5
            },
            IntentCategory.EXCEPTION: {
                "keywords": ["error", "exception", "throw", "catch", "fail", "handling"],
                "patterns": [r'.*error.*', r'.*exception.*', r'.*fail.*', r'.*throw.*'],
                "weight": 1.4
            }
        }

        self.context_limits = {
            IntentCategory.EXCEPTION: {
                "max_tokens": 1200,
                "max_context_chars": 10000,
                "base_nodes": 3,
                "category_nodes": 2,
                "fill_nodes": 1
            },
            IntentCategory.TESTING: {
                "max_tokens": 1000,
                "max_context_chars": 12000,
                "base_nodes": 3,
                "category_nodes": 3,
                "fill_nodes": 2
            },
            IntentCategory.USAGE: {
                "max_tokens": 800,
                "max_context_chars": 8000,
                "base_nodes": 2,
                "category_nodes": 3,
                "fill_nodes": 1
            },
            IntentCategory.DEFINITION: {
                "max_tokens": 400,
                "max_context_chars": 5000,
                "base_nodes": 2,
                "category_nodes": 1,
                "fill_nodes": 0
            },
            IntentCategory.IMPLEMENTATION: {
                "max_tokens": 800,
                "max_context_chars": 6000,
                "base_nodes": 2,
                "category_nodes": 2,
                "fill_nodes": 1
            },
            IntentCategory.GENERAL: {
                "max_tokens": 700,
                "max_context_chars": 4000,
                "base_nodes": 2,
                "category_nodes": 1,
                "fill_nodes": 1
            },
            IntentCategory.MEDIUM: {
                "max_tokens": 500,
                "max_context_chars": 3500,
                "base_nodes": 2,
                "category_nodes": 1,
                "fill_nodes": 0
            },
            IntentCategory.SPECIFIC: {
                "max_tokens": 300,
                "max_context_chars": 2000,
                "base_nodes": 1,
                "category_nodes": 0,
                "fill_nodes": 0
            }
        }

    @property
    def classifier_model(self) -> SentenceTransformer:
        if self._classifier_model is None:
            self._classifier_model = SentenceTransformer(self.classifier_model_name)
        return self._classifier_model

    @property
    def classifier_embeddings(self) -> Dict:
        if self._classifier_embeddings is None:
            try:
                with open(self.classifier_embeddings_path, "r", encoding="utf-8") as f:
                    self._classifier_embeddings = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Classifier embeddings not found at {self.classifier_embeddings_path}")
        return self._classifier_embeddings

    def classify_question_basic(self, question: str) -> str:
        question_emb = self.classifier_model.encode([question], convert_to_tensor=False)[0]
        best_score = -1
        best_label = IntentCategory.GENERAL.value
        for label, examples in self.classifier_embeddings.items():
            for emb in examples:
                score = cosine_similarity([question_emb], [emb])[0][0]
                if score > best_score:
                    best_score = score
                    best_label = label
        return best_label

    def enhanced_classify_question(self, question: str) -> IntentAnalysis:
        basic_category = self.classify_question_basic(question)
        question_lower = question.lower()
        scores = {}
        for category, config in self.intent_patterns.items():
            score = 0.0
            for keyword in config["keywords"]:
                if keyword in question_lower:
                    score += config["weight"]
            for pattern in config["patterns"]:
                if re.search(pattern, question_lower):
                    score += config["weight"] * 1.2
            scores[category.value] = score

        if all(score == 0.0 for score in scores.values()):
            return IntentAnalysis(
                primary_intent=IntentCategory(basic_category),
                secondary_intents=[],
                confidence=0.5,
                scores=scores,
                requires_examples=False,
                requires_usage_info=False,
                requires_implementation_details=False,
                user_expertise_level=ExpertiseLevel.INTERMEDIATE,
                enhanced=False
            )

        best_category = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_category] / total_score if total_score > 0 else 0.5
        secondary_intents = []
        requires_examples = False
        requires_usage_info = False
        requires_implementation_details = False
        if any(word in question_lower for word in ['example', 'sample', 'show me']):
            secondary_intents.append("examples")
            requires_examples = True
        if any(word in question_lower for word in ['best practice', 'recommendation', 'should i']):
            secondary_intents.append("advice")
        if any(word in question_lower for word in ['why', 'reason', 'purpose']):
            secondary_intents.append("reasoning")
        primary_intent_enum = IntentCategory(best_category)
        if primary_intent_enum == IntentCategory.USAGE:
            requires_usage_info = True
        elif primary_intent_enum in [IntentCategory.DEFINITION, IntentCategory.TESTING, IntentCategory.EXCEPTION]:
            requires_examples = True
        elif primary_intent_enum == IntentCategory.IMPLEMENTATION:
            requires_implementation_details = True
        expertise_level = ExpertiseLevel.INTERMEDIATE
        if any(word in question_lower for word in ['simple', 'basic', 'beginner', 'explain like']):
            expertise_level = ExpertiseLevel.BEGINNER
        elif any(word in question_lower for word in ['advanced', 'detailed', 'deep', 'comprehensive']):
            expertise_level = ExpertiseLevel.ADVANCED
        return IntentAnalysis(
            primary_intent=primary_intent_enum,
            secondary_intents=secondary_intents,
            confidence=min(confidence, 1.0),
            scores=scores,
            requires_examples=requires_examples,
            requires_usage_info=requires_usage_info,
            requires_implementation_details=requires_implementation_details,
            user_expertise_level=expertise_level,
            enhanced=True
        )

    def get_context_limits(self, intent: IntentCategory) -> Dict[str, Any]:
        return self.context_limits.get(intent, {
            "max_tokens": 600,
            "max_context_chars": 3000,
            "base_nodes": 2,
            "category_nodes": 1,
            "fill_nodes": 1
        })

    def is_usage_question(self, question: str) -> bool:
        question_lower = question.lower()
        return any(word in question_lower for word in ['used', 'where', 'usage', 'called', 'referenced'])

    def is_description_question(self, question: str) -> bool:
        question_lower = question.lower()
        return any(word in question_lower for word in ['describe', 'how', 'what'])


_intent_analyzer: Optional[IntentAnalyzer] = None


def get_intent_analyzer() -> IntentAnalyzer:
    global _intent_analyzer
    if _intent_analyzer is None:
        _intent_analyzer = IntentAnalyzer()
    return _intent_analyzer


def enhanced_classify_question(question: str) -> Dict[str, Any]:
    analyzer = get_intent_analyzer()
    analysis = analyzer.enhanced_classify_question(question)
    return {
        "category": analysis.primary_intent.value,
        "confidence": analysis.confidence,
        "scores": analysis.scores,
        "enhanced": analysis.enhanced
    }


def classify_question(question: str) -> str:
    analyzer = get_intent_analyzer()
    return analyzer.classify_question_basic(question)


def analyze_user_intent(question: str) -> Dict[str, Any]:
    analyzer = get_intent_analyzer()
    analysis = analyzer.enhanced_classify_question(question)
    return {
        "primary_intent": analysis.primary_intent.value,
        "secondary_intents": analysis.secondary_intents,
        "requires_examples": analysis.requires_examples,
        "requires_usage_info": analysis.requires_usage_info,
        "requires_implementation_details": analysis.requires_implementation_details,
        "user_expertise_level": analysis.user_expertise_level.value
    }
