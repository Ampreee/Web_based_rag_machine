import faiss
import numpy as np
import os
from threading import Lock
from app.core.config import FAISS_INDEX_PATH, EMBED_DIM

_index = None
_next_index_position = 0
_index_lock = Lock()

def _ensure_index():
    global _index, _next_index_position
    with _index_lock:
        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
        
        if _index is None:
            try:
                if os.path.exists(FAISS_INDEX_PATH):
                    print(f"Loading existing FAISS index from {FAISS_INDEX_PATH}")
                    _index = faiss.read_index(FAISS_INDEX_PATH)
                    _next_index_position = _index.ntotal
                    print(f"Loaded index with {_index.ntotal} vectors")
                else:
                    print(f"Creating new FAISS index at {FAISS_INDEX_PATH}")
                    _index = faiss.IndexFlatIP(EMBED_DIM)
                    _next_index_position = 0
                    # Save empty index to ensure we can write to the location
                    save_index()
            except Exception as e:
                print(f"Error with FAISS index: {str(e)}")
                print("Creating new index...")
                _index = faiss.IndexFlatIP(EMBED_DIM)
                _next_index_position = 0
                try:
                    save_index()
                except Exception as e:
                    print(f"Error saving new index: {str(e)}")
                    raise

def save_index():
    global _index
    if _index is None:
        return
    try:
        print(f"Saving FAISS index with {_index.ntotal} vectors to {FAISS_INDEX_PATH}")
        temp_path = f"{FAISS_INDEX_PATH}.tmp"
        faiss.write_index(_index, temp_path)

        os.replace(temp_path, FAISS_INDEX_PATH)
        print("Index saved successfully")
    except Exception as e:
        print(f"Error saving index: {str(e)}")

        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

def add_vectors(vecs: list[list[float]]) -> tuple[int, int]:
    """
    Add vectors to the FAISS index
    Args:
        vecs: list of vectors shape (n, dim)
    Returns:
        tuple[int, int]: (start_idx, count) - starting index and number of vectors added
    """
    global _index, _next_index_position
    _ensure_index()
    
    if not vecs:
        return _next_index_position, 0
        
    with _index_lock:
        try:
            arr = np.array(vecs, dtype='float32')
            if arr.shape[1] != EMBED_DIM:
                raise ValueError(f"Vector dimension {arr.shape[1]} does not match index dimension {EMBED_DIM}")
                
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
            
            start = _next_index_position
            if _index is not None:
                n = arr.shape[0]
                print(f"add_vectors: adding vectors, shape={arr.shape}, EMBED_DIM={EMBED_DIM}")
                add_fn = getattr(_index, "add")
                try:
                    add_fn(np.ascontiguousarray(arr))
                except Exception as e:
                    
                    print(f"FAISS add() raised: {repr(e)}; index type={type(_index)}; index.ntotal={getattr(_index, 'ntotal', None)}")
                    raise
                _next_index_position += n
                save_index()
            return start, arr.shape[0]
        except Exception as e:
            raise RuntimeError(f"Failed to add vectors to index: {str(e)}")

def search(query_vec: list[float], top_k: int = 5) -> tuple[list[float], list[int]]:
    """
    Search for nearest vectors in the index
    Args:
        query_vec: query vector
        top_k: number of results to return
    Returns:
        tuple[list[float], list[int]]: (distances, indices)
    """
    _ensure_index()
    
    if _index is None or _index.ntotal == 0:
        return [], []
        
    with _index_lock:
        try:
            q = np.array([query_vec], dtype='float32')
            if q.shape[1] != EMBED_DIM:
                raise ValueError(f"Query dimension {q.shape[1]} does not match index dimension {EMBED_DIM}")
                
            q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
            
            k = min(top_k, _index.ntotal)

            search_fn = getattr(_index, "search")
            distances, indices = search_fn(np.ascontiguousarray(q), k)
            return distances[0].tolist(), indices[0].tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to search index: {str(e)}")
