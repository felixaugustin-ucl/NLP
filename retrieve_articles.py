import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# Load Articles
# =========================

with open("data/articles.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

texts = [a["Text"] for a in articles]


# =========================
# Build TF-IDF
# =========================

vectorizer = TfidfVectorizer(stop_words="english")

tfidf_matrix = vectorizer.fit_transform(texts)


# =========================
# Retrieval function
# =========================

def retrieve(question, top_k=5):

    q_vec = vectorizer.transform([question])

    scores = cosine_similarity(q_vec, tfidf_matrix)[0]

    ranked = sorted(
        list(enumerate(scores)),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for idx, score in ranked[:top_k]:

        results.append({
            "article": articles[idx]["Article_ID"],
            "score": score,
            "text": articles[idx]["Text"]
        })

    return results


# =========================
# Test
# =========================

query = input("Ask about EU AI Act:\n")

results = retrieve(query)

print("\nTop relevant articles:\n")

for r in results:
    print(f"Article {r['article']}  | score={r['score']:.3f}")
    print(r["text"][:200])
    print()