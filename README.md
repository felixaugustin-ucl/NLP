# Graph Neural Network-Based Retrieval for Grounded Question Answering on the EU AI Act

## Current Pipeline Logic

```text
User Query
  -> Encode query in text embedding space
  -> Dense seed retrieval over node_features (cosine)
  -> 1-hop graph expansion
  -> PCST pruning on expanded subgraph
  -> Build prompt from post-prune node text
  -> Ollama (qwen2.5:3b) generation
  -> Final grounded answer
```

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
- `03_retrieval_prune.ipynb`: query retrieval, k-hop expansion, PCST pruning, and retrieval evaluation.

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

Recommended Python version: `3.11` or above 

### Python Environment Setup (by OS)

- macOS / Linux:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  python -m pip install -r requirements.txt
  ```

- Windows (PowerShell):
  ```powershell
  py -3 -m venv venv
  .\venv\Scripts\Activate.ps1
  python -m pip install -r requirements.txt
  ```

### Docker Setup (Cross-Platform)

If local dependency setup is unstable (especially on Windows), use Docker.

#### Install Docker (by OS)

- Windows:
  - Install Docker Desktop: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
  - Open Docker Desktop once after install and wait until it shows Docker is running.
  - Ensure WSL2 is enabled if Docker Desktop prompts for it.

- macOS (Apple Silicon / Intel):
  - Install Docker Desktop: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
  - Open Docker Desktop and wait until the engine is running.

- Linux:
  - Install Docker Engine: [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)
  - Install Docker Compose plugin: [https://docs.docker.com/compose/install/linux/](https://docs.docker.com/compose/install/linux/)
  - Optional post-install (run Docker without `sudo`): [https://docs.docker.com/engine/install/linux-postinstall/](https://docs.docker.com/engine/install/linux-postinstall/)

Quick check:

```bash
docker --version
docker compose version
```

If both commands return versions, Docker is ready.

#### Build and run this project with Docker

```bash
docker compose build
docker compose run --rm app bash
```

Inside the container, run project commands exactly as usual from `/workspace`.

To run Jupyter from Docker:

```bash
docker compose run --rm -p 8888:8888 app jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Ollama prerequisite:

- Ollama is not installed via `requirements.txt`.
- If Ollama is not installed, `ollama serve` will fail.

### Ollama Installation (by OS)

- macOS (Apple users):
  - Install from [Ollama](https://ollama.com/download).
  - Then run:
    ```bash
    ollama serve
    ```

- Linux users:
  - Install with:
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```
  - Then run:
    ```bash
    ollama serve
    ```

- Windows users:
  - Install from [Ollama](https://ollama.com/download).
  - Then run in PowerShell:
    ```powershell
    ollama serve
    ```

After Ollama is running, pull the model:

```bash
ollama pull qwen2.5:3b
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

Final retrieval + LLM answer:

```bash
# start Ollama in one terminal
ollama serve

# in another terminal
ollama pull qwen2.5:3b
python3 scripts/retrieval_prune_cli.py
```

If you run the script inside Docker while Ollama runs on your host machine, use:

```bash
OLLAMA_BASE_URL=http://host.docker.internal:11434 python3 scripts/retrieval_prune_cli.py
```

Notes:

- `scripts/retrieval_prune_cli.py` follows the same retrieval/prune logic as `notebooks/03_retrieval_prune.ipynb`.
- The script uses post-prune nodes as context for `scripts/prompts.py`.
- The script calls local Ollama at `http://localhost:11434` with model `qwen2.5:3b`.
- Output is the generated answer (not pre/post-prune tables).
