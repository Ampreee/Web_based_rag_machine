from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from arq import create_pool
from groq import Groq
from arq.connections import RedisSettings
from app.core.config import REDIS_DSN, TOP_K, GROQ_API
import aiosqlite
from app.db.sqlite_store import SQLITE_PATH
import asyncio
from typing import Optional

router = APIRouter()

class QueryRequest(BaseModel):
    q: str

def get_groq_response(prompt: str) -> str:
    """Get response from Groq API with simplified streaming."""
    if not GROQ_API:
        return "LLM not configured. Set GROQ_API to enable synthesis."
    
    client = Groq(api_key=GROQ_API)
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{
            "role": "system",
            "content": "You are a precise and accurate summarizer. Your task is to:"
                      "\n1. Read the given context carefully"
                      "\n2. Focus only on information directly relevant to the question"
                      "\n3. Extract factual information and specific details"
                      "\n4. Provide accurate numbers, dates, and statistics when present"
                      "\n5. Do not make assumptions or add information not present in the context"
                      "\n6. Present information in a clear, structured manner"
                      "\n7. If the context doesn't contain enough information to fully answer the question, acknowledge this"
        },
        {
            "role": "user", 
            "content": prompt
        }],
        temperature=0.3,  
        max_completion_tokens=8192,
        top_p=0.9,
        stream=True
    )
    
    response_text = []
    for chunk in completion:
        if hasattr(chunk.choices[0].delta, 'content'):
            if chunk.choices[0].delta.content:
                response_text.append(chunk.choices[0].delta.content)
    
    return "".join(response_text)

async def get_context_chunks(vector_ids: list) -> list:
    """Retrieve text chunks from SQLite database."""
    async with aiosqlite.connect(SQLITE_PATH) as db:
        placeholders = ",".join("?" for _ in vector_ids)
        query = f"SELECT text, vector_id FROM chunk WHERE vector_id IN ({placeholders})"
        cur = await db.execute(query, vector_ids)
        rows = await cur.fetchall()
        return [{"text": row[0], "vector_id": row[1]} for row in rows]

@router.post("/")
async def query(req: QueryRequest):
    if not REDIS_DSN:
        raise HTTPException(status_code=500, detail="REDIS_DSN environment variable is not set")

    pool = None
    try:
        pool = await create_pool(RedisSettings.from_dsn(str(REDIS_DSN)))
        job = await pool.enqueue_job("search_topk", req.q, TOP_K)
        if job is None:
            raise HTTPException(status_code=500, detail="Failed to enqueue search job")

        results = await job.result(timeout=60)
        if not results:
            raise HTTPException(status_code=404, detail="No relevant documents found")

        vector_ids = [r["vector_id"] for r in results if r.get("vector_id")]
        chunks = await get_context_chunks(vector_ids) if vector_ids else []

        context_text = "\n\n".join(f"[{i+1}] {chunk['text']}" for i, chunk in enumerate(chunks))
        prompt = f"""Analyze the provided context chunks and create a comprehensive answer that addresses the question. Structure your response in this exact format:

Question: {req.q}

your Response should start from below:-

[Provide the main analysis and facts here without citations. Include specific numbers, dates, and statistics.]

Key Points:
• [Point 1]
• [Point 2]
• [Point 3]
...

Sources:
[At the end, list all citations in format: [X] where X is the chunk number from which information was drawn]

Context chunks:
{context_text}

Guidelines:
1. Keep citations ONLY in the "Sources" section at the end
2. Include specific numbers and data in the main analysis
3. Present information objectively and accurately
4. Use bullet points for key findings
5. Order citations numerically
6. If information is insufficient or conflicting, acknowledge this in the analysis

Answer:"""
        answer = await asyncio.to_thread(get_groq_response, prompt)
        return {"answer": answer}

    except TimeoutError:
        raise HTTPException(status_code=504, detail="Search operation timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if pool:
            await pool.close()
