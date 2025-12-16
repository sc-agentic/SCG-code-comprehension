from typing import Optional
from loguru import logger
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()


def judge_answer(prompt: str) -> Optional[int]:

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}])
        model_response = message.content[0].text.strip()
        logger.debug(f"Judge response: {model_response}")
        for char in model_response:
            if char.isdigit() and char in "012345":
                return int(char)
        return None
    except Exception as e:
        logger.error(f"Judge error: {e}")
        return None