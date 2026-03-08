import os
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.mixture import GaussianMixture

# Absolute Path Logic
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "vector_storage")

def run_indexing():
    print(f"🚀 Initializing Indexer at: {DB_PATH}")
    
    # 1. Fetch & Clean Data
    print("📦 Downloading UCI Newsgroups...")
    data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    docs = [d.strip() for d in data.data if len(d.strip()) > 100][:1000] 

    # 2. Embedding
    print("🧠 Generating Embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(docs, show_progress_bar=True)

    # 3. Fuzzy Clustering (GMM)
    print("🖇️ Calculating Fuzzy Membership (GMM)...")
    gmm = GaussianMixture(n_components=20, random_state=42)
    gmm.fit(embeddings)
    probs = gmm.predict_proba(embeddings)

    # 4. Storage
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name="forum_collection")
    
    collection.add(
        documents=docs,
        embeddings=embeddings.tolist(),
        metadatas=[{"fuzzy_scores": str(p.tolist())} for p in probs],
        ids=[f"id_{i}" for i in range(len(docs))]
    )
    print(f"✅ Success! {collection.count()} documents indexed.")

if __name__ == "__main__":
    run_indexing()