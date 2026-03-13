# Graph Neural Network-Based Retrieval for Grounded Question Answering on the EU AI Act

## Project Overview

This project builds a system that lets users ask questions about the **EU AI Act** and receive answers **grounded in the document itself**, reducing hallucination risk. It uses a Graph Neural Network (GNN) to learn structure-aware embeddings of legal provisions.

### Core Research Question

> Can a GNN trained on the structure and cross-references of the EU AI Act improve retrieval quality over text-only retrieval for grounded legal question answering?

## Architecture

```
User Query → Query Encoder → GNN-Enhanced Retrieval → Context Expansion → Grounded Answer Generation
```

### Pipeline Stages

1. **PDF Parsing** – Convert EU AI Act PDF to structured text
2. **Node Construction** – Split into legal chunks (articles, paragraphs, recitals, annexes, definitions)
3. **Edge Construction** – Build relationships (part_of, refers_to, defines, uses_term, similar_to)
4. **Node Feature Generation** – Text embeddings + metadata features
5. **GNN Training** – Self-supervised contrastive learning on graph
6. **Retrieval** – Cosine similarity between query and graph-aware embeddings
7. **Grounded QA** – Answer generation with citations from retrieved evidence

## Project Structure

```
nlp_group_project/
├── configs/                    # Configuration files
│   └── config.yaml             # Main project configuration
├── data/
│   ├── raw/                    # Original EU AI Act PDF
│   ├── processed/              # Parsed text, nodes.csv, edges.csv
│   ├── embeddings/             # Text embeddings (pre-GNN)
│   └── graphs/                 # PyG graph objects
├── src/
│   ├── parsing/                # PDF parsing and text extraction
│   ├── graph_construction/     # Node and edge construction
│   ├── features/               # Feature engineering (embeddings + metadata)
│   ├── gnn/                    # GNN model definition and training
│   ├── retrieval/              # Retrieval pipeline
│   └── qa/                     # Grounded answer generation
├── notebooks/                  # Jupyter notebooks for exploration
├── models/
│   ├── checkpoints/            # Saved model weights
│   └── embeddings/             # Final graph-aware node embeddings
├── evaluation/
│   ├── results/                # Evaluation metrics and tables
│   └── plots/                  # Figures and visualisations
├── scripts/                    # Utility and runner scripts
├── tests/                      # Unit tests
└── docs/                       # Documentation, diagrams, report drafts
```

## Setup

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric
- Transformers (HuggingFace)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd nlp_group_project

# Create virtual environment (requires Python 3.11)
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (handles PyTorch Geometric in correct order)
bash install.sh
```

> **Note:** Do not use `pip install -r requirements.txt` directly — the PyTorch Geometric packages require a special installation order handled by `install.sh`.

## Usage

```bash
# 1. Parse the EU AI Act PDF
python -m src.parsing.pdf_parser

# 2. Construct graph (nodes + edges)
python -m src.graph_construction.build_graph

# 3. Generate node features
python -m src.features.generate_features

# 4. Train the GNN
Run `notebooks/02_gnn_training.ipynb` (instead of `python -m src.gnn.train`)

# 5. Run retrieval evaluation
python -m evaluation.evaluate_retrieval

# 6. Run QA demo
python -m src.qa.generate_answer --query "What are the obligations of providers of high-risk AI systems?"
```

## Evaluation

| Method | Recall@5 | Recall@10 | MRR | NDCG@10 |
|--------|----------|-----------|-----|---------|
| Text-only retrieval | – | – | – | – |
| Graph heuristic (PageRank) | – | – | – | – |
| **GraphSAGE (ours)** | – | – | – | – |

## Team

- **GNN Module**: [Your Name]
- PDF Parsing: [Team Member]
- QA Generation: [Team Member]

## License

This project is for academic purposes (UCL NLP course).
