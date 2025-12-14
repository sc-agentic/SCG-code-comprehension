from transformers import AutoTokenizer
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_tokenizer():
    return AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    tokenizer = _get_tokenizer()
    return len(tokenizer.encode(text))