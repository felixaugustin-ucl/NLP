import pandas as pd
import re
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities

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
# Community detection
# =========================

communities = list(greedy_modularity_communities(G.to_undirected()))

colors = [
    "#4A90E2",
    "#50E3C2",
    "#F5A623",
    "#D0021B",
    "#7ED321",
    "#9013FE"
]

color_map = {}

for i, community in enumerate(communities):
    for node in community:
        color_map[node] = colors[i % len(colors)]

node_colors = [color_map[node] for node in G.nodes()]

# =========================
# Draw graph
# =========================

plt.figure(figsize=(16,12))

pos = nx.spring_layout(G, k=0.5, seed=42)

# draw nodes
nx.draw_networkx_nodes(
    G,
    pos,
    node_color=node_colors,
    node_size=800,
    alpha=0.9
)

# draw edges WITH ARROWS
nx.draw_networkx_edges(
    G,
    pos,
    arrows=True,
    arrowstyle='-|>',
    arrowsize=15,
    edge_color="gray",
    width=1.5,
    alpha=0.6
)

# draw labels
nx.draw_networkx_labels(
    G,
    pos,
    font_size=8
)

plt.title("EU AI Act Knowledge Graph (GraphRAG)", fontsize=18)

plt.axis("off")

plt.tight_layout()

plt.savefig("eu_ai_act_graph.png", dpi=300)

plt.show()