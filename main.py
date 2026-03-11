import pandas as pd
import re
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Load dataset
# =========================

df = pd.read_csv("data/eu_law_dataset.csv")

texts = df["Text"].tolist()

# =========================
# Build Graph
# =========================

G = nx.DiGraph()

for article in df["Article_ID"]:
    G.add_node(article)

pattern = r'Article\s+(\d+)'

for index, row in df.iterrows():

    source_article = row["Article_ID"]
    text = row["Text"]

    refs = re.findall(pattern, text)

    for ref in refs:

        target = "Article " + ref

        if target != source_article and target in df["Article_ID"].values:
            G.add_edge(source_article, target)

# =========================
# Load embedding model
# =========================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings...")

embeddings = model.encode(texts)

# =========================
# User query
# =========================

query = input("\nEnter your legal query: ")

query_embedding = model.encode([query])

similarity = cosine_similarity(query_embedding, embeddings)[0]

top_indices = similarity.argsort()[-5:][::-1]

print("\nTop semantic articles:\n")

top_articles = []

for i in top_indices:

    article_id = df.iloc[i]["Article_ID"]
    article_text = df.iloc[i]["Text"]

    print(article_id)
    print(article_text[:200])
    print()

    top_articles.append(article_id)

# =========================
# Graph expansion
# =========================

print("\nGraph-related articles:\n")

neighbors = set()

for article in top_articles:

    for neighbor in G.neighbors(article):
        neighbors.add(neighbor)

for n in list(neighbors)[:5]:

    row = df[df["Article_ID"] == n]

    print(n)
    print(row["Text"].values[0][:200])
    print()