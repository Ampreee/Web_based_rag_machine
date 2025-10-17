def chunk_text(text: str, max_chars: int = 1800, overlap: int = 300):
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    idx = 0
    while start < L:
        end = min(L, start + max_chars)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == L:
            break
        start = max(0, end - overlap)
        idx += 1
    return chunks
