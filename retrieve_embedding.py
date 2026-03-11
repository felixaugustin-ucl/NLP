import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# load articles
with open("data/articles.json","r",encoding="utf-8") as f:
    articles = json.load(f)

texts = [a["Text"] for a in articles]

# load embeddings
embeddings = np.load("embeddings/article_embeddings.npy")

def retrieve(query, top_k=5):

    query_embedding = model.encode([query])

    scores = cosine_similarity(query_embedding, embeddings)[0]

    top_idx = scores.argsort()[-top_k:][::-1]

    results = []

    for idx in top_idx:

        results.append({
            "article": articles[idx]["Article_ID"],
            "score": float(scores[idx]),
            "text": articles[idx]["Text"][:300]
        })

    return results