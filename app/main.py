import os
import time
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from .cache_system import SemanticCache

app = FastAPI(title="NeuralForum AI Engine")

# --- Model & DB Initialization ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "vector_storage")

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection("forum_collection")
cache = SemanticCache()

# --- Statistics State ---
class PerformanceTracker:
    def __init__(self):
        self.hit_count = 0
        self.miss_count = 0

    @property
    def hit_rate(self):
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0

stats = PerformanceTracker()

class QueryRequest(BaseModel):
    query: str

# --- Helper Function for Metadata ---
def parse_probs(raw_meta):
    """Safely converts metadata string to list if necessary."""
    if isinstance(raw_meta, str):
        try:
            return eval(raw_meta)
        except:
            return [0.0] * 20 # Fallback
    return raw_meta

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def get_index():
    path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(path, "r") as f:
        return f.read()

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    q_vec = model.encode(request.query)
    
    # 1. Check Semantic Cache
    cached_res, sim = cache.get(q_vec)
    if cached_res:
        stats.hit_count += 1
        probs = parse_probs(cached_res[0]['fuzzy_probabilities'])
        return {
            "source": "semantic_cache",
            "similarity": float(sim),
            "results": cached_res,
            "dominant_cluster": int(np.argmax(probs))
        }
    
    # 2. Database Search (Cache Miss)
    stats.miss_count += 1
    results = collection.query(query_embeddings=[q_vec.tolist()], n_results=3)
    
    output = []
    for i in range(len(results['documents'][0])):
        output.append({
            "content": results['documents'][0][i],
            "fuzzy_probabilities": results['metadatas'][0][i]
        })
    
    # 3. Update Cache
    cache.set(q_vec, output)
    probs = parse_probs(output[0]['fuzzy_probabilities'])
    
    return {
        "source": "database",
        "results": output,
        "dominant_cluster": int(np.argmax(probs))
    }

@app.get("/cache/stats")
async def get_cache_stats():
    return {
        "total_entries": len(cache.cache),
        "hit_count": stats.hit_count,
        "miss_count": stats.miss_count,
        "hit_rate": f"{stats.hit_rate:.2%}"
    }

@app.delete("/cache")
async def clear_cache():
    cache.cache.clear()
    stats.hit_count = 0
    stats.miss_count = 0
    return {"status": "success", "message": "Cache flushed"}