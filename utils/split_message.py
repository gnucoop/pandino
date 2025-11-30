# utils/split_message.py

"""
Utility function for splitting long messages into WhatsApp-sized chunks.
"""

def split_message(text: str, limit: int = 900) -> list[str]:
    """
    Splits a long text into chunks of at most `limit` characters,
    without breaking words. Adds (i/n) prefix to each chunk.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current_words: list[str] = []
    length = 0

    for w in words:
        extra = len(w) + (1 if current_words else 0)
        if length + extra > limit:
            chunks.append(" ".join(current_words))
            current_words = [w]
            length = len(w)
        else:
            current_words.append(w)
            length += extra

    if current_words:
        chunks.append(" ".join(current_words))

    total = len(chunks)
    return [f"({i}/{total}) {c}" for i, c in enumerate(chunks, start=1)]
