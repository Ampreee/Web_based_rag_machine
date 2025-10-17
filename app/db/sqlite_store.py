import aiosqlite
import os
from app.core.config import SQLITE_PATH, MAPPING_TABLE

async def init_db():
    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
    async with aiosqlite.connect(SQLITE_PATH) as db:
        await db.execute(f"""
        CREATE TABLE IF NOT EXISTS ingestion_job (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'pending',
            num_urls INTEGER DEFAULT 0,
            submitted_by TEXT
        )""")
        await db.execute(f"""
        CREATE TABLE IF NOT EXISTS source (
            id TEXT PRIMARY KEY,
            job_id TEXT,
            url TEXT UNIQUE,
            status TEXT DEFAULT 'pending',
            fetched_at TIMESTAMP,
            http_status INTEGER,
            error TEXT
        )""")
        await db.execute(f"""
        CREATE TABLE IF NOT EXISTS chunk (
            id TEXT PRIMARY KEY,
            source_id TEXT,
            vector_id TEXT,
            chunk_index INTEGER,
            text TEXT,
            metadata TEXT
        )""")
        await db.execute(f"""
        CREATE TABLE IF NOT EXISTS {MAPPING_TABLE} (
            vector_id TEXT PRIMARY KEY,
            vector_idx INTEGER, -- position index inside faiss
            url TEXT
        )""")
        await db.commit()

async def insert_ingestion_job(db, job_id, urls, submitted_by=None):
    await db.execute("INSERT OR IGNORE INTO ingestion_job (id, num_urls, submitted_by) VALUES (?, ?, ?)",
                    (job_id, len(urls), submitted_by))
    for u in urls:
        await db.execute("INSERT OR IGNORE INTO source (id, job_id, url, status) VALUES (?, ?, ?, ?)",
                        (u, job_id, u, "pending")) 
    await db.commit()

async def update_source_status(db, url, status, http_status=None, fetched_at=None, error=None):
    await db.execute("UPDATE source SET status=?, http_status=?, fetched_at=?, error=? WHERE url=?",
                    (status, http_status, fetched_at, error, url))
    await db.commit()

async def insert_chunks(db, url, chunks, vector_ids, vector_idxs):
    source_id = url
    for i, (text, vid, vidx) in enumerate(zip(chunks, vector_ids, vector_idxs)):
        await db.execute(
            "INSERT INTO chunk (id, source_id, vector_id, chunk_index, text, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (f"{url}::chunk::{i}", source_id, vid, i, text, "{}")
        )
        await db.execute(
            f"INSERT OR REPLACE INTO {MAPPING_TABLE} (vector_id, vector_idx, url) VALUES (?, ?, ?)",
            (vid, vidx, url)
        )
    await db.commit()

async def fetch_chunks_by_vector_ids(db, vector_ids):
    placeholders = ",".join("?" for _ in vector_ids)
    q = f"SELECT text, vector_id FROM chunk WHERE vector_id IN ({placeholders})"
    rows = await db.execute(q, vector_ids)
    cur = await db.execute(q, vector_ids)
    res = await cur.fetchall()
    return [{"text": r[0], "vector_id": r[1]} for r in res]
