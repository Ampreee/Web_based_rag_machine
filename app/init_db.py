import asyncio
from app.db.sqlite_store import init_db
asyncio.run(init_db())
print("DB initialized")
