import json
import csv
import os
import re

# =========================
# Load Articles
# =========================

with open("data/articles.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

print("Loaded articles:", len(articles))
print("Example article:", articles[0])

nodes = {}
edges = []

# =========================
# Simple rule extraction
# =========================

for art in articles:

    # 兼容不同字段名
    article_id = art.get("Article_ID") or art.get("Article") or art.get("article") or art.get("id")
    text = art.get("Text") or art.get("text")

    if article_id is None or text is None:
        continue

    # 提取数字编号
    article_num = re.findall(r"\d+", str(article_id))
    if article_num:
        article_num = article_num[0]
    else:
        article_num = str(article_id)

    node_id = f"A{article_num}"

    # =========================
    # Create Article Node
    # =========================

    nodes[node_id] = {
        "id": node_id,
        "label": f"Article {article_num}",
        "type": "Article"
    }

    # =========================
    # Relation Detection
    # =========================

    text_lower = text.lower()

    if "prohibit" in text_lower:
        relation = "prohibits"

    elif "require" in text_lower:
        relation = "requires"

    elif "define" in text_lower:
        relation = "defines"

    else:
        relation = "mentions"

    # =========================
    # Extract Concepts
    # =========================

    concepts = re.findall(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)*", text)

    for concept in concepts[:3]:

        cid = concept.strip()

        if len(cid) < 3:
            continue

        nodes[cid] = {
            "id": cid,
            "label": cid,
            "type": "Concept"
        }

        edges.append({
            "source": node_id,
            "target": cid,
            "relation": relation
        })

# =========================
# Create graph folder
# =========================

os.makedirs("graph", exist_ok=True)

# =========================
# Save nodes.csv
# =========================

with open("graph/nodes.csv", "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=["id", "label", "type"]
    )

    writer.writeheader()

    for node in nodes.values():
        writer.writerow(node)

# =========================
# Save edges.csv
# =========================

with open("graph/edges.csv", "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=["source", "target", "relation"]
    )

    writer.writeheader()

    for edge in edges:
        writer.writerow(edge)

# =========================
# Done
# =========================

print("Knowledge Graph created successfully")
print("Total Nodes:", len(nodes))
print("Total Edges:", len(edges))
print("Saved to folder: graph/")