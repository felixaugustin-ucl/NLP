"""
Simple retrieval module for the EU AI Act project.

Supports:
- Text-only retrieval with sentence embeddings
- GNN retrieval with graph-aware embeddings
- Optional 1-hop context expansion for GNN retrieval
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return v / norm


@dataclass
class RetrievedNode:
    rank: int
    node_id: str
    score: float
    text: str
    type: str
    expanded: bool = False

    def as_dict(self) -> dict:
        return {
            "rank": self.rank,
            "node_id": self.node_id,
            "score": self.score,
            "text": self.text,
            "type": self.type,
            "expanded": self.expanded,
        }


class BaseRetriever:
    def __init__(self, nodes_df: pd.DataFrame, embeddings: np.ndarray):
        if nodes_df.empty:
            raise ValueError("nodes_df is empty")
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")

        self.nodes_df = nodes_df.reset_index(drop=True)
        self.embeddings = np.asarray(_normalize(embeddings), dtype=np.float32)
        self.node_ids = self.nodes_df["node_id"].astype(str).tolist()
        self.id_to_text = dict(
            zip(self.node_ids, self.nodes_df.get("text", pd.Series([""] * len(self.nodes_df))).fillna("").astype(str).tolist())
        )
        self.id_to_type = dict(
            zip(self.node_ids, self.nodes_df.get("type", pd.Series([""] * len(self.nodes_df))).astype(str).tolist())
        )
        self._id_to_idx = {node_id: i for i, node_id in enumerate(self.node_ids)}

        if len(self.node_ids) != len(self.embeddings):
            raise ValueError("Number of node IDs and embeddings do not match")

        self.encoder = None

    def encode_query(self, query: str) -> np.ndarray:
        if self.encoder is None:
            raise RuntimeError("Encoder is not initialised")
        q = self.encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        return _normalize(np.asarray(q, dtype=np.float32))

    def _score_all(self, query_emb: np.ndarray) -> np.ndarray:
        scores = cosine_similarity(query_emb, self.embeddings)[0]
        return scores.astype(np.float32)

    def retrieve(self, query: str, top_k: int = 10) -> List[dict]:
        if top_k <= 0:
            return []

        query = str(query).strip()
        if not query:
            return []

        q = self.encode_query(query)
        scores = self._score_all(q)
        order = np.argsort(scores)[::-1][:top_k]

        out = []
        for i, idx in enumerate(order.tolist()):
            node_id = self.node_ids[int(idx)]
            out.append(
                RetrievedNode(
                    rank=i + 1,
                    node_id=node_id,
                    score=float(scores[int(idx)]),
                    text=self.id_to_text[node_id],
                    type=self.id_to_type[node_id],
                ).as_dict()
            )
        return out


class TextRetriever(BaseRetriever):
    def __init__(
        self,
        nodes_df: pd.DataFrame,
        text_embeddings: np.ndarray,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        super().__init__(nodes_df, text_embeddings)
        self.encoder = SentenceTransformer(model_name)


class GNNRetriever(BaseRetriever):
    def __init__(
        self,
        nodes_df: pd.DataFrame,
        gnn_embeddings: np.ndarray,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        graph_data=None,
        text_embeddings: Optional[np.ndarray] = None,
    ):
        super().__init__(nodes_df, gnn_embeddings)
        self.encoder = SentenceTransformer(model_name)
        self.graph_data = graph_data
        self._query_projection = None

        if text_embeddings is not None:
            self._fit_query_projection(text_embeddings, gnn_embeddings)

    def _fit_query_projection(self, text_emb: np.ndarray, gnn_emb: np.ndarray):
        if text_emb.shape[0] != gnn_emb.shape[0]:
            logger.warning(
                "Cannot learn query projection: text_embeddings and gnn_embeddings must have same number of nodes."
            )
            return
        try:
            W, _, _, _ = np.linalg.lstsq(text_emb, gnn_emb, rcond=None)
            self._query_projection = W
            logger.info(f"Learned query projection: {W.shape[0]} -> {W.shape[1]}")
        except Exception as exc:
            logger.warning("Query projection fit failed, using raw query embedding: %s", exc)

    def encode_query(self, query: str) -> np.ndarray:
        q = super().encode_query(query)
        if self._query_projection is None:
            return q
        return _normalize(q @ self._query_projection)

    def expand_context(self, results: List[dict], hops: int = 1) -> List[dict]:
        if hops != 1:
            logger.info("Only 1-hop expansion is implemented in this simple version.")
        if self.graph_data is None or not hasattr(self.graph_data, "edge_index"):
            logger.warning("No graph_data.edge_index available for context expansion.")
            return results

        edge_index = self.graph_data.edge_index
        if torch.is_tensor(edge_index):
            edge_index = edge_index.detach().cpu().numpy()
        edge_index = np.asarray(edge_index)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            logger.warning("Unexpected edge_index shape for expansion: %s", edge_index.shape)
            return results

        result_node_ids = [r["node_id"] for r in results]
        expanded = []
        expanded_set = set(result_node_ids)

        for node_id in result_node_ids:
            if node_id not in self._id_to_idx:
                continue
            src = self._id_to_idx[node_id]
            nbr_mask = edge_index[0] == src
            nbrs = edge_index[1][nbr_mask].astype(int).tolist()
            for n in nbrs:
                if 0 <= n < len(self.node_ids):
                    nid = self.node_ids[n]
                    if nid not in expanded_set:
                        expanded_set.add(nid)
                        expanded.append(
                            {
                                "rank": len(result_node_ids) + len(expanded) + 1,
                                "node_id": nid,
                                "score": 0.0,
                                "text": self.id_to_text[nid],
                                "type": self.id_to_type[nid],
                                "expanded": True,
                            }
                        )

        return results + expanded


def print_results(results: List[dict], max_text_len: int = 180):
    print(f"\n{'=' * 70}")
    print(f"Retrieved {len(results)} nodes")
    print(f"{'=' * 70}")
    for r in results:
        preview = r["text"][:max_text_len]
        if len(r["text"]) > max_text_len:
            preview += "..."
        mark = " [expanded]" if r.get("expanded") else ""
        print(f"[{r['rank']}] {r['node_id']}{mark} | score={r['score']:.4f}")
        print(f"    type: {r['type']}")
        print(f"    text: {preview}\n")


def main():
    parser = argparse.ArgumentParser(description="Run simple retrieval")
    parser.add_argument("--query", required=True, type=str)
    parser.add_argument("--method", choices=["text", "gnn"], default="gnn")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--expand", action="store_true", help="expand GNN results with neighbors")
    args = parser.parse_args()

    config = load_config()
    nodes_df = pd.read_csv(config["paths"]["nodes_csv"])

    if args.method == "text":
        text_emb = np.load(config["paths"]["node_features"])
        retriever = TextRetriever(nodes_df, text_emb, config["embedding"]["model_name"])
        results = retriever.retrieve(args.query, top_k=args.top_k)
    else:
        gnn_emb = np.load(config["paths"]["final_embeddings"])
        graph = torch.load(config["paths"]["graph_object"], weights_only=False)
        retriever = GNNRetriever(
            nodes_df=nodes_df,
            gnn_embeddings=gnn_emb,
            model_name=config["embedding"]["model_name"],
            graph_data=graph,
            text_embeddings=np.load(config["paths"]["node_features"]),
        )
        results = retriever.retrieve(args.query, top_k=args.top_k)
        if args.expand:
            results = retriever.expand_context(results, hops=1)

    print_results(results)


if __name__ == "__main__":
    main()
