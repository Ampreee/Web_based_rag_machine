from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api import ingest, query
from app.db.sqlite_store import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    pass

app = FastAPI(title="Web RAG (FAISS + SQLite)", lifespan=lifespan)

app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(query.router, prefix="/query", tags=["query"])

@app.get("/")
async def root():
    return {"message": "Web RAG (FAISS+SQLite) ready. top_k=5"}
