"""
Retrieval Pipeline.

Implements:
1. Text-only baseline retrieval (cosine similarity on raw embeddings)
2. GNN-enhanced retrieval (cosine similarity on graph-aware embeddings)
3. Context expansion (1-hop neighbor retrieval)

Usage:
    python -m src.retrieval.retriever --query "What are the obligations of high-risk AI providers?"
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class BaseRetriever:
    """Base class for retrievers."""

    def __init__(self, nodes_df: pd.DataFrame, embeddings: np.ndarray):
        self.nodes_df = nodes_df
        self.embeddings = embeddings
        self.node_ids = nodes_df["node_id"].tolist()

    def encode_query(self, query: str) -> np.ndarray:
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        raise NotImplementedError


class TextRetriever(BaseRetriever):
    """
    Baseline: text-only retrieval using sentence embeddings.

    No graph information is used.
    """

    def __init__(
        self,
        nodes_df: pd.DataFrame,
        text_embeddings: np.ndarray,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        super().__init__(nodes_df, text_embeddings)
        self.encoder = SentenceTransformer(model_name)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encoder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Retrieve top-k nodes by cosine similarity with query.
        """
        query_emb = self.encode_query(query)
        similarities = cosine_similarity(query_emb, self.embeddings)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "rank": len(results) + 1,
                "node_id": self.node_ids[idx],
                "score": float(similarities[idx]),
                "text": self.nodes_df.iloc[idx].get("text", ""),
                "type": self.nodes_df.iloc[idx].get("type", ""),
            })

        return results


class GNNRetriever(BaseRetriever):
    """
    GNN-enhanced retrieval using graph-aware node embeddings.

    Uses the same query encoder but compares against
    GNN-produced node embeddings.
    """

    def __init__(
        self,
        nodes_df: pd.DataFrame,
        gnn_embeddings: np.ndarray,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        graph_data=None,
    ):
        super().__init__(nodes_df, gnn_embeddings)
        self.encoder = SentenceTransformer(model_name)
        self.graph_data = graph_data

        # Project query to GNN embedding space
        # TODO: optionally learn a projection layer
        # For now, we use a simple linear projection
        self.query_projection = None

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query using text encoder."""
        emb = self.encoder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )
        # TODO: Apply projection to GNN embedding space if trained
        return emb

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """Retrieve top-k nodes using GNN embeddings."""
        query_emb = self.encode_query(query)
        similarities = cosine_similarity(query_emb, self.embeddings)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "rank": len(results) + 1,
                "node_id": self.node_ids[idx],
                "score": float(similarities[idx]),
                "text": self.nodes_df.iloc[idx].get("text", ""),
                "type": self.nodes_df.iloc[idx].get("type", ""),
            })

        return results

    def expand_context(
        self, results: list[dict], hops: int = 1
    ) -> list[dict]:
        """
        Expand retrieved results by including 1-hop neighbors.

        This adds linked definitions, annexes, and parent articles
        to improve answer completeness.
        """
        if self.graph_data is None:
            logger.warning("No graph data available for context expansion")
            return results

        expanded_ids = set(r["node_id"] for r in results)
        edge_index = self.graph_data.edge_index.numpy()
        id_to_idx = {nid: i for i, nid in enumerate(self.node_ids)}
        idx_to_id = {i: nid for nid, i in id_to_idx.items()}

        for r in results:
            node_idx = id_to_idx.get(r["node_id"])
            if node_idx is None:
                continue

            # Find neighbors
            mask = edge_index[0] == node_idx
            neighbor_indices = edge_index[1][mask]

            for n_idx in neighbor_indices:
                n_id = idx_to_id.get(int(n_idx))
                if n_id and n_id not in expanded_ids:
                    expanded_ids.add(n_id)
                    idx = id_to_idx[n_id]
                    results.append({
                        "rank": len(results) + 1,
                        "node_id": n_id,
                        "score": 0.0,  # expansion, no direct score
                        "text": self.nodes_df.iloc[idx].get("text", ""),
                        "type": self.nodes_df.iloc[idx].get("type", ""),
                        "expanded": True,
                    })

        return results


def print_results(results: list[dict], max_text_len: int = 200):
    """Pretty-print retrieval results."""
    print(f"\n{'='*80}")
    print(f"Retrieved {len(results)} nodes:")
    print(f"{'='*80}")
    for r in results:
        text_preview = r["text"][:max_text_len] + "..." if len(r["text"]) > max_text_len else r["text"]
        expanded = " [EXPANDED]" if r.get("expanded") else ""
        print(f"\n  [{r['rank']}] {r['node_id']} (score: {r['score']:.4f}){expanded}")
        print(f"      Type: {r['type']}")
        print(f"      Text: {text_preview}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--method", choices=["text", "gnn"], default="gnn")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    config = load_config()

    nodes_df = pd.read_csv(config["paths"]["nodes_csv"])

    if args.method == "text":
        text_embeddings = np.load(config["paths"]["node_features"])
        retriever = TextRetriever(
            nodes_df, text_embeddings, config["embedding"]["model_name"]
        )
    else:
        gnn_embeddings = np.load(config["paths"]["final_embeddings"])
        graph_data = torch.load(config["paths"]["graph_object"])
        retriever = GNNRetriever(
            nodes_df, gnn_embeddings, config["embedding"]["model_name"],
            graph_data=graph_data
        )

    results = retriever.retrieve(args.query, top_k=args.top_k)

    if args.method == "gnn" and config["retrieval"]["context_expansion"]["enabled"]:
        results = retriever.expand_context(
            results, hops=config["retrieval"]["context_expansion"]["hops"]
        )

    print_results(results)


if __name__ == "__main__":
    main()
