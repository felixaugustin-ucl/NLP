import pandas as pd
import re
import networkx as nx

# =========================
# Load dataset
# =========================

df = pd.read_csv("data/eu_law_dataset.csv")

# =========================
# Build Graph
# =========================

G = nx.DiGraph()

for article in df["Article_ID"]:
    G.add_node(article)

pattern = r'Article\s+(\d+)'

for index, row in df.iterrows():

    source = row["Article_ID"]
    text = row["Text"]

    refs = re.findall(pattern, text)

    for ref in refs:

        target = "Article " + ref

        if target != source and target in df["Article_ID"].values:
            G.add_edge(source, target)

# =========================
# Print ASCII Graph
# =========================

print("\nEU AI Act GraphRAG Text Graph\n")

for node in G.nodes():

    neighbors = list(G.neighbors(node))

    if neighbors:

        print(node)

        for i, n in enumerate(neighbors):

            if i == len(neighbors)-1:
                print("   └──→", n)
            else:
                print("   ├──→", n)

        print()
        # =========================
# Print Text Graph (ASCII)
# =========================

print("\nEU AI Act GraphRAG Text Graph\n")

for node in G.nodes():

    neighbors = list(G.neighbors(node))

    if neighbors:

        print(node)

        for i, n in enumerate(neighbors):

            if i == len(neighbors)-1:
                print("   └──→", n)
            else:
                print("   ├──→", n)

        print()