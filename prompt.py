from typing import Dict, Any
from models import ConversationHistory
from intent_analyzer import get_intent_analyzer, IntentCategory


def build_intent_aware_prompt(question: str, context: str, intent: Dict[str, Any], conversation_history: ConversationHistory) -> str:
    primary_intent = intent["primary_intent"]
    try:
        intent_category = IntentCategory(primary_intent)
    except ValueError:
        intent_category = IntentCategory.GENERAL
    if intent_category == IntentCategory.USAGE:
        task = "Analyze the provided code to identify where and how the requested code element is used.\n" \
               "Focus on:\n" \
               "1. Classes and methods that call or reference it\n" \
               "2. Specific code lines showing its usage\n" \
               "3. Usage context (controllers, services, tests, repositories, etc.)\n" \
               "4. HTTP endpoints or API mappings that utilize it\n" \
               "\n" \
               "Important: Code sections are organized with '## TYPE: Name' headers for easy navigation.\n" \
               "If test classes or methods are included in the code below, please reference them in your answer.\n" \
               "Base your response strictly on the code provided."
    elif intent_category == IntentCategory.DEFINITION:
        task = "Describe the code element based on the provided code.\n" \
               "\n" \
               "Include:\n" \
               "- What it is and its primary purpose\n" \
               "- Visible fields, methods, and annotations\n" \
               "- Structure and key characteristics\n" \
               "\n" \
               "Focus only on what is visible in the code below."
    else:
        task_definitions = {
            IntentCategory.EXCEPTION: "Find error handling and exception management in this code. Base your answer ONLY on what you see in the provided code.",
            IntentCategory.IMPLEMENTATION: "Explain how this code works internally based only on the provided code. Do not add general knowledge.",
            IntentCategory.TESTING: "Analyze the testing approach and test coverage based only on the test code provided.",
            IntentCategory.GENERAL: "Analyze the provided code based only on what you see.",
            IntentCategory.MEDIUM: "Analyze the provided code with moderate detail based only on what you see.",
            IntentCategory.SPECIFIC: "Provide specific technical details about this code based only on what you see."
        }
        task = task_definitions.get(intent_category, "Analyze the provided code based only on what you see.")

    analyzer = get_intent_analyzer()
    limits = analyzer.get_context_limits(intent_category)
    max_context_chars = limits["max_context_chars"]

    if len(context) > max_context_chars:
        if '##' in context:
            sections = context.split('\n\n##')
            truncated = []
            current_length = 0
            for i, section in enumerate(sections):
                if i == 0:
                    section_text = section
                else:
                    section_text = '##' + section
                if current_length + len(section_text) + 2 <= max_context_chars:
                    truncated.append(section_text)
                    current_length += len(section_text) + 2
                else:
                    break

            if truncated:
                context = '\n\n'.join(truncated)
                if len(truncated) < len(sections):
                    context += f"\n\n... [{len(sections) - len(truncated)} sections omitted]"
            else:
                context = context[:max_context_chars] + "\n... [truncated]"
        else:
            context = context[:max_context_chars] + "\n... [truncated]"
    prompt_parts = [
        f"INSTRUCTIONS: {task}",
        "",
        "CODE TO ANALYZE:",
        context.strip(),
        "",
        f"USER QUESTION: {question}",
        "",
        "ANALYSIS:",
        "Based on the code sections above:"
    ]

    if intent_category not in [IntentCategory.EXCEPTION, IntentCategory.TESTING]:
        history_context = conversation_history.get_conversation_context()
        if history_context and len(history_context) < 500:
            prompt_parts.insert(-3, f"Previous conversation: {history_context[:300]}")

    final_prompt = "\n".join(prompt_parts)
    print(f"Context length: {len(context)} chars")
    print(f"Has ## headers: {context.count('##')} sections")
    print(f"Contains 'testUpdateWebinar': {'testUpdateWebinar' in context}")
    print(f"Final prompt length: {len(final_prompt)} chars")

    return final_prompt


def post_process_answer(answer: str, intent: Dict[str, Any]) -> str:
    processed_answer = answer.strip()
    if len(processed_answer) < 50:
        processed_answer += "\n\nNo detailed information found in the provided code."
    primary_intent = intent.get("primary_intent", "general")
    try:
        intent_category = IntentCategory(primary_intent)
    except ValueError:
        intent_category = IntentCategory.GENERAL
    if intent.get("requires_examples", False) and intent_category in [IntentCategory.DEFINITION, IntentCategory.TESTING]:
        if "example" not in processed_answer.lower() and len(processed_answer) < 200:
            processed_answer += "\n\nNote: For more specific examples, please provide additional context or ask about particular use cases."

    if intent.get("requires_usage_info", False) and intent_category == IntentCategory.USAGE:
        if "used" not in processed_answer.lower() and "usage" not in processed_answer.lower():
            processed_answer += "\n\nNote: Usage information may be limited based on the provided code context."
        if "test" in processed_answer.lower() and "test" not in processed_answer.lower():
            processed_answer += "\n\nNote: This method is also used in test classes (see test code for examples)."

    if intent.get("requires_implementation_details", False) and intent_category == IntentCategory.IMPLEMENTATION:
        if len(processed_answer) < 100:
            processed_answer += "\n\nNote: Implementation details may require access to more comprehensive code context."

    expertise_level = intent.get("user_expertise_level", "intermediate")
    if expertise_level == "beginner" and len(processed_answer) > 50:
        technical_terms = ["API", "interface", "implementation", "polymorphism", "abstraction"]
        if any(term in processed_answer for term in technical_terms):
            if "note:" not in processed_answer.lower():
                processed_answer += "\n\nNote: This explanation contains technical terms. Feel free to ask for clarification on any concepts."

    return processed_answer
