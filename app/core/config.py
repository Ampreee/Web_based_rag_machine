import os
from dotenv import load_dotenv
load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./data")
os.makedirs(DATA_DIR, exist_ok=True)  
SQLITE_PATH = os.path.join(DATA_DIR, "data.db")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
MAPPING_TABLE = "vector_map" 
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384")) 
TOP_K = int(os.getenv("TOP_K", "5"))
_raw_redis = os.getenv("REDIS_DSN") or ""
REDIS_DSN = _raw_redis.strip().strip('"').strip("'")
if not REDIS_DSN:
    raise ValueError("REDIS_DSN environment variable must be set with your Upstash Redis URL")
GROQ_API= os.getenv("GROQ_API", "")
