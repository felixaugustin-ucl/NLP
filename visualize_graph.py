import pandas as pd
import re
import networkx as nx
import matplotlib.pyplot as plt

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
# Draw Graph
# =========================

plt.figure(figsize=(14,10))

pos = nx.spring_layout(G, k=0.5)

nx.draw_networkx_nodes(
    G, pos,
    node_size=700,
    node_color="skyblue"
)

nx.draw_networkx_edges(
    G, pos,
    arrows=True,
    edge_color="gray"
)

nx.draw_networkx_labels(
    G, pos,
    font_size=8
)

plt.title("EU AI Act Knowledge Graph", fontsize=18)

plt.axis("off")

plt.show()