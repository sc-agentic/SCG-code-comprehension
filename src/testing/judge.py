import re

import anthropic
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

client = anthropic.Anthropic()


def clean_json(response_text: str) -> str:
    pattern = r"```(?:json)?\s*(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL)

    if match:
        return match.group(1).strip()
    return response_text.strip()

def judge_answer(prompt: str) -> str:

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}])
        model_response = message.content[0].text.strip()
        clean_response = clean_json(model_response)
        logger.debug(f"Judge response: {clean_response}")
        return clean_response
    except Exception as e:
        logger.error(f"Judge error: {e}")
        return None