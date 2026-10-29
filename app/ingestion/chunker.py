from dataclasses import dataclass


@dataclass
class TextChunk:
    text: str
    page_number: int
    chunk_index: int


def chunk_text(
    text: str,
    page_number: int,
    chunk_size: int = 180,
    overlap: int = 30,
) -> list[TextChunk]:
    """
    Split text into word-based chunks.

    chunk_size and overlap are measured in words, not model tokens.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    words = text.split()
    chunks = []
    start = 0
    chunk_index = 0
    step = chunk_size - overlap

    while start < len(words):
        piece = words[start : start + chunk_size]

        if piece:
            chunks.append(
                TextChunk(
                    text=" ".join(piece),
                    page_number=page_number,
                    chunk_index=chunk_index,
                )
            )

        start += step
        chunk_index += 1

    return chunks
