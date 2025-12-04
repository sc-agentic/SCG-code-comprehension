CODE_CHARS_PER_TOKEN = 3.5
TEXT_CHARS_PER_TOKEN = 4.2


def estimate_tokens(text: str, is_code: bool = True) -> int:
    """
    Estimates token count for a text snippet.

    Args:
        text: Input text to estimate
        is_code: Whether the input is source code (default: True)

    Returns:
        Estimated token count (minimum 1)
    """
    if not text:
        return 0
    if is_code:
        chars_per_token = CODE_CHARS_PER_TOKEN
    else:
        chars_per_token = TEXT_CHARS_PER_TOKEN
    return max(1, int(len(text) / chars_per_token))
