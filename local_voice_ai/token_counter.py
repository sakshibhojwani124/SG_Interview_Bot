from transformers import AutoTokenizer

# Change if you're using another Gemma model
tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-2-2b"
)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))