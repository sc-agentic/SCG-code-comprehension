from transformers import AutoTokenizer
from functools import lru_cache
import tiktoken


@lru_cache(maxsize=1)
def _get_llama_tokenizer():
    return AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")


@lru_cache(maxsize=1)
def _get_gpt5_tokenizer():
    return tiktoken.encoding_for_model("gpt-4o-mini")


def count_tokens(text: str, model: str = "llama") -> int:
    if not text:
        return 0

    if model.lower() == "llama":
        tokenizer = _get_llama_tokenizer()
        return len(tokenizer.encode(text))

    elif model.lower() == "gpt5":
        if _get_gpt5_tokenizer() is None:
            raise ImportError("Failed to import GPT tokenizer")
        tokenizer = _get_gpt5_tokenizer()
        return len(tokenizer.encode(text))

    else:
        raise KeyError("Unsupported model type")
