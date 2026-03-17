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
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
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


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", str(text).lower())


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

    def retrieve_from_embedding(self, query_emb: np.ndarray, top_k: int = 10) -> List[dict]:
        if top_k <= 0:
            return []
        scores = self._score_all(_normalize(np.asarray(query_emb, dtype=np.float32)))
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

    def retrieve(self, query: str, top_k: int = 10) -> List[dict]:
        query = str(query).strip()
        if not query:
            return []
        q = self.encode_query(query)
        return self.retrieve_from_embedding(q, top_k=top_k)


class TextRetriever(BaseRetriever):
    def __init__(
        self,
        nodes_df: pd.DataFrame,
        text_embeddings: np.ndarray,
        model_name: str | None = None,
        encoder: Optional[SentenceTransformer] = None,
    ):
        super().__init__(nodes_df, text_embeddings)
        if encoder is None and not model_name:
            raise ValueError("model_name must be provided when encoder is not supplied.")
        self.encoder = encoder if encoder is not None else SentenceTransformer(model_name)


class BM25Retriever:
    def __init__(self, nodes_df: pd.DataFrame):
        if nodes_df.empty:
            raise ValueError("nodes_df is empty")

        self.nodes_df = nodes_df.reset_index(drop=True)
        self.node_ids = self.nodes_df["node_id"].astype(str).tolist()
        texts = self.nodes_df.get("text", pd.Series([""] * len(self.nodes_df))).fillna("").astype(str).tolist()
        self.id_to_text = dict(zip(self.node_ids, texts))
        self.id_to_type = dict(
            zip(self.node_ids, self.nodes_df.get("type", pd.Series([""] * len(self.nodes_df))).astype(str).tolist())
        )
        self.corpus_tokens = [_tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def retrieve(self, query: str, top_k: int = 10) -> List[dict]:
        if top_k <= 0:
            return []

        query = str(query).strip()
        if not query:
            return []

        scores = np.asarray(self.bm25.get_scores(_tokenize(query)), dtype=np.float32)
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


def _minmax_normalize_dict(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    values = np.asarray(list(scores.values()), dtype=np.float32)
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: float((v - lo) / (hi - lo)) for k, v in scores.items()}


def main_retrieval_test(
    query: str,
    nodes_df: pd.DataFrame,
    text_embeddings: np.ndarray,
    model_name: str | None = None,
    top_k: int = 10,
    seed_k: int = 5,
    graph_data=None,
    gnn_embeddings: Optional[np.ndarray] = None,
    expand_hops: int = 1,
    max_expand: int = 10,
    bm25_weight: float = 0.35,
    dense_weight: float = 0.40,
    gnn_weight: float = 0.25,
    cross_encoder: Optional[CrossEncoder] = None,
    rerank_weight: float = 0.40,
    shared_encoder: Optional[SentenceTransformer] = None,
) -> List[dict]:
    query = str(query).strip()
    if not query:
        return []
    if shared_encoder is None and not model_name:
        raise ValueError("model_name must be provided when shared_encoder is not supplied.")

    bm25_retriever = BM25Retriever(nodes_df)
    dense_retriever = TextRetriever(nodes_df, text_embeddings, model_name=model_name, encoder=shared_encoder)
    gnn_retriever = None
    if gnn_embeddings is not None:
        gnn_retriever = GNNRetriever(
            nodes_df=nodes_df,
            gnn_embeddings=gnn_embeddings,
            model_name=model_name,
            graph_data=graph_data,
            text_embeddings=text_embeddings,
            encoder=shared_encoder,
        )

    bm25_results = bm25_retriever.retrieve(query, top_k=top_k)
    dense_query_emb = dense_retriever.encode_query(query)
    dense_results = dense_retriever.retrieve_from_embedding(dense_query_emb, top_k=top_k)

    bm25_scores = _minmax_normalize_dict({r["node_id"]: float(r["score"]) for r in bm25_results})
    dense_scores = _minmax_normalize_dict({r["node_id"]: float(r["score"]) for r in dense_results})

    merged_ids = set(bm25_scores) | set(dense_scores)
    merged = []
    for node_id in merged_ids:
        merged.append(
            {
                "node_id": node_id,
                "bm25_score": bm25_scores.get(node_id, 0.0),
                "dense_score": dense_scores.get(node_id, 0.0),
                "merge_score": (
                    bm25_weight * bm25_scores.get(node_id, 0.0)
                    + dense_weight * dense_scores.get(node_id, 0.0)
                ),
            }
        )
    merged.sort(key=lambda row: row["merge_score"], reverse=True)

    top_seed_ids = [row["node_id"] for row in merged[: max(0, int(seed_k))]]
    expanded_ids = []
    if gnn_retriever is not None and top_seed_ids:
        seed_rows = []
        for rank, node_id in enumerate(top_seed_ids, start=1):
            seed_rows.append(
                {
                    "rank": rank,
                    "node_id": node_id,
                    "score": next((row["merge_score"] for row in merged if row["node_id"] == node_id), 0.0),
                    "text": gnn_retriever.id_to_text.get(node_id, ""),
                    "type": gnn_retriever.id_to_type.get(node_id, ""),
                    "expanded": False,
                }
            )
        expanded = gnn_retriever.expand_context(
            seed_rows,
            hops=expand_hops,
            seed_k=seed_k,
            max_expanded=max_expand,
        )
        expanded_ids = [row["node_id"] for row in expanded if row.get("expanded")]

    gnn_scores: Dict[str, float] = {}
    if gnn_retriever is not None:
        candidate_ids = list(dict.fromkeys(top_seed_ids + expanded_ids + [row["node_id"] for row in merged[:top_k]]))
        if candidate_ids:
            q_gnn = gnn_retriever.project_query_embedding(dense_query_emb)
            candidate_indices = [gnn_retriever._id_to_idx[node_id] for node_id in candidate_ids if node_id in gnn_retriever._id_to_idx]
            if candidate_indices:
                sims = cosine_similarity(q_gnn, gnn_retriever.embeddings[candidate_indices])[0]
                gnn_scores = _minmax_normalize_dict(
                    {
                        gnn_retriever.node_ids[idx]: float(score)
                        for idx, score in zip(candidate_indices, sims.tolist())
                    }
                )

    final_ids = list(dict.fromkeys([row["node_id"] for row in merged[:top_k]] + expanded_ids))
    cross_scores: Dict[str, float] = {}
    if cross_encoder is not None and final_ids:
        pairs = []
        pair_ids = []
        for node_id in final_ids:
            row = nodes_df.loc[nodes_df["node_id"].astype(str) == node_id]
            if row.empty:
                continue
            text = str(row.iloc[0].get("text", ""))
            if not text.strip():
                continue
            pairs.append((query, text))
            pair_ids.append(node_id)
        if pairs:
            raw_cross_scores = np.asarray(cross_encoder.predict(pairs), dtype=np.float32).reshape(-1)
            cross_scores = _minmax_normalize_dict(
                {node_id: float(score) for node_id, score in zip(pair_ids, raw_cross_scores.tolist())}
            )

    reranked = []
    node_lookup = nodes_df.set_index(nodes_df["node_id"].astype(str))
    for node_id in final_ids:
        if node_id not in node_lookup.index:
            continue
        merged_score = next((row["merge_score"] for row in merged if row["node_id"] == node_id), 0.0)
        expanded_bonus = 0.05 if node_id in expanded_ids else 0.0
        base_score = (
            merged_score
            + gnn_weight * gnn_scores.get(node_id, 0.0)
            + expanded_bonus
        )
        final_score = (
            (1.0 - rerank_weight) * base_score
            + rerank_weight * cross_scores.get(node_id, 0.0)
        )
        row = node_lookup.loc[node_id]
        reranked.append(
            {
                "node_id": node_id,
                "score": float(final_score),
                "base_score": float(base_score),
                "bm25_score": float(bm25_scores.get(node_id, 0.0)),
                "dense_score": float(dense_scores.get(node_id, 0.0)),
                "gnn_score": float(gnn_scores.get(node_id, 0.0)),
                "cross_score": float(cross_scores.get(node_id, 0.0)),
                "text": str(row.get("text", "")),
                "type": str(row.get("type", "")),
                "expanded": node_id in expanded_ids,
            }
        )

    reranked.sort(key=lambda row: row["score"], reverse=True)
    out = []
    for rank, row in enumerate(reranked[:top_k], start=1):
        out.append(
            {
                "rank": rank,
                "node_id": row["node_id"],
                "score": row["score"],
                "text": row["text"],
                "type": row["type"],
                "expanded": row["expanded"],
                "base_score": row["base_score"],
                "bm25_score": row["bm25_score"],
                "dense_score": row["dense_score"],
                "gnn_score": row["gnn_score"],
                "cross_score": row["cross_score"],
            }
        )
    return out


class GNNRetriever(BaseRetriever):
    def __init__(
        self,
        nodes_df: pd.DataFrame,
        gnn_embeddings: np.ndarray,
        model_name: str | None = None,
        graph_data=None,
        text_embeddings: Optional[np.ndarray] = None,
        encoder: Optional[SentenceTransformer] = None,
    ):
        super().__init__(nodes_df, gnn_embeddings)
        if encoder is None and not model_name:
            raise ValueError("model_name must be provided when encoder is not supplied.")
        self.encoder = encoder if encoder is not None else SentenceTransformer(model_name)
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
        return self.project_query_embedding(q)

    def project_query_embedding(self, query_emb: np.ndarray) -> np.ndarray:
        q = _normalize(np.asarray(query_emb, dtype=np.float32))
        if self._query_projection is None:
            return q
        return _normalize(q @ self._query_projection)

    def expand_context(
        self,
        results: List[dict],
        hops: int = 1,
        seed_k: int = 5,
        max_expanded: int = 10,
    ) -> List[dict]:
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

        edge_confidence = None
        if hasattr(self.graph_data, "edge_confidence"):
            edge_confidence = self.graph_data.edge_confidence
            if torch.is_tensor(edge_confidence):
                edge_confidence = edge_confidence.detach().cpu().numpy()
            edge_confidence = np.asarray(edge_confidence).reshape(-1)

        seed_results = results[: max(0, int(seed_k))]
        result_node_ids = [r["node_id"] for r in results]
        expanded_set = set(result_node_ids)
        candidate_scores: dict[str, float] = {}

        for base_rank, seed in enumerate(seed_results, start=1):
            node_id = seed["node_id"]
            if node_id not in self._id_to_idx:
                continue
            node_idx = self._id_to_idx[node_id]
            out_mask = edge_index[0] == node_idx
            in_mask = edge_index[1] == node_idx
            match_positions = np.flatnonzero(out_mask | in_mask).tolist()
            trust_boost = 1.0 / float(base_rank)

            for pos in match_positions:
                src_idx = int(edge_index[0, pos])
                dst_idx = int(edge_index[1, pos])
                nbr_idx = dst_idx if src_idx == node_idx else src_idx
                if not (0 <= nbr_idx < len(self.node_ids)):
                    continue
                nbr_id = self.node_ids[nbr_idx]
                if nbr_id in expanded_set:
                    continue

                conf = 1.0
                if edge_confidence is not None and pos < len(edge_confidence):
                    try:
                        conf = float(edge_confidence[pos])
                    except Exception:
                        conf = 1.0

                candidate_score = float(seed["score"]) * conf * trust_boost
                if candidate_score > candidate_scores.get(nbr_id, float("-inf")):
                    candidate_scores[nbr_id] = candidate_score

        ranked_neighbors = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
        if max_expanded is not None and max_expanded >= 0:
            ranked_neighbors = ranked_neighbors[: int(max_expanded)]
        expanded = []
        for offset, (nbr_id, nbr_score) in enumerate(ranked_neighbors, start=1):
            expanded.append(
                {
                    "rank": len(results) + offset,
                    "node_id": nbr_id,
                    "score": float(nbr_score),
                    "text": self.id_to_text[nbr_id],
                    "type": self.id_to_type[nbr_id],
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
    parser.add_argument("--method", choices=["bm25", "dense", "text", "gnn"], default="gnn")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--expand", action="store_true", help="expand GNN results with neighbors")
    args = parser.parse_args()

    config = load_config()
    nodes_df = pd.read_csv(config["paths"]["nodes_csv"])

    if args.method == "bm25":
        retriever = BM25Retriever(nodes_df)
        results = retriever.retrieve(args.query, top_k=args.top_k)
    elif args.method in {"text", "dense"}:
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
