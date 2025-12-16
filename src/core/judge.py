from typing import Optional
from loguru import logger
import anthropic
from dotenv import load_dotenv

load_dotenv()

JUDGE_PROMPT = """You are a strict judge. Your task is to 
evaluate the answer to the question based only on the provided source context.

Judging rules:
1. Use only the information present in the source context.
2. Do not use external knowledge or assume missing details.

Input:
source context: {context}
question: {question}
answer: {answer}

Scoring scale:
5 — Fully correct, entirely supported by the context, no hallucinations  
4 — Correct overall
3 — Partially correct, some unsupported claims  
2 — Mostly incorrect
1 — Incorrect or contains hallucinations

Output: Return only integer from 1 to 5."""

client = anthropic.Anthropic()


def judge_answer(question: str, answer: str, context: str) -> Optional[int]:

    prompt = JUDGE_PROMPT.format(
        context=context,
        question=question,
        answer=answer)
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}])
        model_response = message.content[0].text.strip()
        for char in model_response:
            if char.isdigit() and char in "12345":
                return int(char)
        return None
    except Exception as e:
        logger.error(f"Judge error: {e}")
        return None