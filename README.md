# 🧠 NeuralForum: AI-Powered Semantic Search Engine

NeuralForum is an industry-grade search engine that utilizes **Sentence Transformers**, **FastAPI**, and **Fuzzy Clustering (GMM)** to provide context-aware results. It features a custom **Semantic LRU Cache** to drastically reduce latency for similar queries.

## 🚀 Key Features
- **Semantic Vector Search:** Uses `all-MiniLM-L6-v2` for high-dimensional embeddings.
- **Semantic Cache:** Custom `OrderedDict` implementation for sub-10ms response times on cached intents.
- **Fuzzy Clustering:** Implements **Gaussian Mixture Models (GMM)** for probabilistic topic assignment.
- **Performance Dashboard:** Real-time monitoring of Cache Hit Rate and Query Source.

## 🛠️ Installation & Setup
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/forum_engine.git](https://github.com/YOUR_USERNAME/forum_engine.git)
   cd forum_engine
