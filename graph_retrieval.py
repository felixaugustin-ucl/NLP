import pandas as pd

# =========================
# Load Graph
# =========================

nodes = pd.read_csv("graph/nodes.csv")
edges = pd.read_csv("graph/edges.csv")

# =========================
# Graph Expansion Function
# =========================

def expand_graph(article_id):

    article_node = f"A{article_id}"

    related = edges[edges["source"] == article_node]

    results = []

    for _, row in related.iterrows():

        concept = row["target"]
        relation = row["relation"]

        results.append({
            "concept": concept,
            "relation": relation
        })

    return results


# =========================
# Test
# =========================

if __name__ == "__main__":

    article = 5

    connections = expand_graph(article)

    print("Article", article, "connections:")

    for c in connections:
        print(c)