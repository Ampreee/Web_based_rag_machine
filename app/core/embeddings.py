from sentence_transformers import SentenceTransformer
import asyncio
from app.core.config import EMBED_MODEL

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

async def embed_texts(texts: list[str]) -> list[list[float]]:
    model=_get_model()
    embeddings = await asyncio.to_thread(model.encode, texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
    return embeddings.tolist()
async def embed_query(text: str) -> list[float]:
    vecs = await embed_texts([text])
    return vecs[0]  
