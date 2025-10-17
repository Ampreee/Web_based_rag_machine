import aiosqlite
import os
from app.core.config import SQLITE_PATH, MAPPING_TABLE

import aiosqlite
import os

SQLITE_PATH = os.getenv("SQLITE_PATH", "./db.sqlite3")

async def init_db():
    async with aiosqlite.connect(SQLITE_PATH) as db:
        await db.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_jobs (
            job_id TEXT PRIMARY KEY,
            url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        await db.execute("""
        CREATE TABLE IF NOT EXISTS sources (
            url TEXT PRIMARY KEY,
            status TEXT,
            http_status INTEGER,
            fetched_at TEXT,
            error TEXT
        )
        """)
        await db.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            chunk TEXT,
            vector_id TEXT,
            vector_idx INTEGER
        )
        """)
        await db.execute("""
        CREATE TABLE IF NOT EXISTS vector_map (
            vector_id TEXT PRIMARY KEY,
            url TEXT,
            vector_idx INTEGER
        )
        """)
        await db.commit()

async def insert_ingestion_job(job_id, url):
    async with aiosqlite.connect(SQLITE_PATH) as db:
        await db.execute("INSERT INTO ingestion_jobs (job_id, url) VALUES (?, ?)", (job_id, url))
        await db.commit()

async def update_source_status(db, url, status, http_status=None, fetched_at=None, error=None):
    await db.execute("REPLACE INTO sources (url, status, http_status, fetched_at, error) VALUES (?, ?, ?, ?, ?)", (url, status, http_status, fetched_at, error))
    await db.commit()

async def insert_chunks(db, url, chunks, vector_ids, vector_idxs):
    for chunk, vid, vidx in zip(chunks, vector_ids, vector_idxs):
        await db.execute("INSERT INTO chunks (url, chunk, vector_id, vector_idx) VALUES (?, ?, ?, ?)", (url, chunk, vid, vidx))
        await db.execute("INSERT INTO vector_map (vector_id, url, vector_idx) VALUES (?, ?, ?)", (vid, url, vidx))
    await db.commit()

async def fetch_chunks_by_vector_ids(db, vector_ids):
    placeholders = ",".join("?" for _ in vector_ids)
    q = f"SELECT text, vector_id FROM chunk WHERE vector_id IN ({placeholders})"
    rows = await db.execute(q, vector_ids)
    cur = await db.execute(q, vector_ids)
    res = await cur.fetchall()
    return [{"text": r[0], "vector_id": r[1]} for r in res]
