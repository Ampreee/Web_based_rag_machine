from fastapi import APIRouter
from pydantic import BaseModel, HttpUrl
import uuid
from arq.connections import RedisSettings
from app.db.sqlite_store import init_db, SQLITE_PATH
import aiosqlite
from arq import create_pool
from app.core.config import REDIS_DSN

router = APIRouter()

class IngestRequest(BaseModel):
    urls: list[HttpUrl]
    submitted_by: str | None = None

@router.post("/", status_code=202)
async def ingest(req: IngestRequest):
    job_id = str(uuid.uuid4())
    await init_db()
    async with aiosqlite.connect(SQLITE_PATH) as db:
        await db.execute("INSERT OR IGNORE INTO ingestion_job (id, num_urls, submitted_by) VALUES (?, ?, ?)",
                        (job_id, len(req.urls), req.submitted_by))
        for u in req.urls:
            await db.execute("INSERT OR IGNORE INTO source (id, job_id, url, status) VALUES (?, ?, ?, ?)", (str(u), job_id, str(u), "queued"))
        await db.commit()
    if not REDIS_DSN:
        raise ValueError("REDIS_DSN environment variable is not set")
    redis_settings = RedisSettings.from_dsn(REDIS_DSN)
    pool = await create_pool(redis_settings)
    try:
        for u in req.urls:
            await pool.enqueue_job("process_url", job_id, str(u))
    finally:
        await pool.close()
    return {"job_id": job_id, "queued": len(req.urls)}
