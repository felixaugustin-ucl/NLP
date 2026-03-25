# Graph Neural Network-Based Retrieval for Grounded Question Answering on the EU AI Act

## 1) Notebook Workflow

Run notebooks in this order:

1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_gcn_training.ipynb`
3. `notebooks/02_gnn_training.ipynb`
4. `notebooks/03_retrieval_prune.ipynb`

What each notebook does:

- `01_data_exploration.ipynb`: data/graph inspection and sanity checks.
- `02_gcn_training.ipynb`: trains GCN and writes `models/embeddings/gcn_node_embeddings.npy`.
- `02_gnn_training.ipynb`: trains GraphSAGE/GNN variant and writes `models/embeddings/gnn_node_embeddings.npy`.
- `03_retrieval_prune.ipynb`: query retrieval, k-hop expansion, and PCST pruning over the graph.

## 2) Script Pipeline (Project Structure + How To Run)

Relevant structure:

```text
configs/config.yaml
data/raw/eu_ai_act.pdf
data/processed/{parsed_text.json,nodes.csv,edges.csv}
data/embeddings/node_features.npy
data/graphs/eu_ai_act_graph.pt
src/parsing/pdf_parser.py
src/graph_construction/{node_builder.py,edge_builder.py,build_graph.py}
src/features/generate_features.py
scripts/prompts.py
scripts/retrieval_prune_cli.py
```

Setup:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run full pipeline from repo root:

```bash
# 1) PDF -> parsed text
python -m src.parsing.pdf_parser

# 2) parsed text -> nodes.csv
python -m src.graph_construction.node_builder

# 3) nodes.csv -> edges.csv
python -m src.graph_construction.edge_builder

# 4) nodes.csv -> node_features.npy
python -m src.features.generate_features

# 5) nodes.csv + edges.csv + features -> graph .pt
python -m src.graph_construction.build_graph
```

Final retrieval + LLM answer (uses post-prune nodes as prompt context):

```bash
# start Ollama in one terminal
ollama serve

# in another terminal
ollama pull qwen2.5:3b
python3 scripts/retrieval_prune_cli.py
```

Notes:

- `scripts/retrieval_prune_cli.py` follows the same retrieval/prune logic as `notebooks/03_retrieval_prune.ipynb`.
- It does not print pre/post tables now; it sends post-prune node text to `scripts/prompts.py`, then calls Ollama (`qwen2.5:3b`) and prints the final answer.
