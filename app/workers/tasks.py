import aiohttp
import asyncio
import uuid
from datetime import datetime,UTC
from arq.connections import RedisSettings
from app.core.config import TOP_K
from app.db.sqlite_store import init_db, insert_ingestion_job, update_source_status, insert_chunks, SQLITE_PATH
from app.utils.html_parser import extract_text
from app.core.chunker import chunk_text
from app.core.embeddings import embed_texts, embed_query
from app.core.faiss_client import add_vectors, search, save_index
import aiosqlite
import os
import numpy as np
from app.core.config import REDIS_DSN

async def fetch_html(session, url, timeout=20):
    async with session.get(url, timeout=timeout, headers={"User-Agent":"web-rag-bot/1.0"}) as resp:
        text = await resp.text()
        return resp.status, text, resp.headers.get("content-length")

async def process_url(ctx, job_id: str, url: str):
    await init_db()
    async with aiosqlite.connect(SQLITE_PATH) as db:
        await update_source_status(db, url, "fetching")
        try:
            async with aiohttp.ClientSession() as session:
                status, html, content_len = await fetch_html(session, url)
        except Exception as e:
            await update_source_status(db, url, "failed", None, None, str(e))
            return {"status": "failed", "error": str(e)}

        fetched_at = datetime.now(UTC).isoformat()
        await update_source_status(db, url, "fetched", status, fetched_at, None)

        text = extract_text(html)
        if not text or len(text.strip()) < 50:
            await update_source_status(db, url, "failed", status, fetched_at, "no_text")
            return {"status": "failed", "error": "no_text"}

        chunks = chunk_text(text)
        if not chunks:
            await update_source_status(db, url, "failed", status, fetched_at, "no_chunks")
            return {"status": "failed", "error": "no_chunks"}

        embeddings_np = await embed_texts(chunks)  

        vecs = []
        for v in embeddings_np:
            if isinstance(v, np.ndarray):
                vecs.append(v.astype('float32').tolist())
            else:
                vecs.append([float(x) for x in v])
        start_idx, count = add_vectors(vecs)

        vector_ids = [str(uuid.uuid4()) for _ in range(count)]
        vector_idxs = list(range(start_idx, start_idx + count))

        await insert_chunks(db, url, chunks, vector_ids, vector_idxs)
        await update_source_status(db, url, "embedded", status, fetched_at, None)

        save_index()
        return {"status": "ok", "vectors": count}

async def search_topk(ctx, query: str, top_k: int = TOP_K):
    qvec = await embed_query(query)
    scores, idxs = search(qvec, top_k)
    await init_db()
    async with aiosqlite.connect(SQLITE_PATH) as db:
        rows = []
        for idx in idxs:
            cur = await db.execute("SELECT vector_id, url FROM vector_map WHERE vector_idx = ?", (idx,))
            r = await cur.fetchone()
            rows.append(r)
    results = []
    for r, s in zip(rows, scores):
        if r:
            results.append({"vector_id": r[0], "url": r[1], "score": float(s)})
    return results

class WorkerSettings:
    redis_settings = RedisSettings.from_dsn(REDIS_DSN)
    functions = [process_url, search_topk]
    max_jobs = 8 
