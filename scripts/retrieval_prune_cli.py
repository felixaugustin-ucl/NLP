#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from urllib import error, request

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from pcst_fast import pcst_fast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from scripts.prompts import build_prompt
except ImportError:
    from prompts import build_prompt

NUM_RETRIEVED_SEEDS = 7
K_HOPS = 1
PRIZE_TOP_K = 10
EDGE_COST = 1.0


def resolve_repo_root() -> Path:
    here = Path.cwd()
    if (here / "configs").exists():
        return here
    if (here / ".." / "configs").exists():
        return (here / "..").resolve()
    return here


REPO_ROOT = resolve_repo_root()


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def build_graph(edges_df: pd.DataFrame, node_ids: list[str] | None = None) -> nx.Graph:
    graph = nx.Graph()
    if node_ids is not None:
        for node_id in node_ids:
            graph.add_node(node_id)
    for row in edges_df.itertuples(index=False):
        if row.source != row.target:
            graph.add_edge(row.source, row.target)
    return graph


def get_text_rows(node_ids: list[str], nodes_df: pd.DataFrame) -> pd.DataFrame:
    requested = [str(node_id) for node_id in node_ids]
    order = {node_id: idx for idx, node_id in enumerate(requested)}
    matches = nodes_df[nodes_df["node_id"].isin(requested)].copy()
    matches["request_order"] = matches["node_id"].map(order)
    cols = ["node_id", "title", "type", "text"]
    return matches.sort_values("request_order")[cols].reset_index(drop=True)


def encode_query(query: str, query_encoder: SentenceTransformer, embedding_dim: int) -> np.ndarray:
    query_emb = query_encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    if int(query_emb.shape[0]) != int(embedding_dim):
        raise ValueError(
            f"Query embedding dim {query_emb.shape[0]} != node embedding dim {embedding_dim}. "
            "Check config['embedding']['model_name'] vs node_features generation model."
        )
    return np.asarray(query_emb, dtype=float)


def retrieve_seed_nodes(query_emb: np.ndarray, node_embeds: np.ndarray, node_ids: list[str], k: int = 2) -> tuple[list[str], np.ndarray]:
    similarities = cosine_similarity([query_emb], node_embeds)[0]
    ranked_idx = np.argsort(similarities)[::-1]
    top_idx = ranked_idx[: min(k, len(ranked_idx))]
    return [node_ids[idx] for idx in top_idx], similarities


def expand_k_hops(graph: nx.Graph, seeds: list[str], k: int = 2) -> nx.Graph:
    valid_seeds = [seed for seed in seeds if seed in graph]
    nodes_in_scope = set(valid_seeds)
    frontier = set(valid_seeds)
    for _ in range(k):
        nxt: set[str] = set()
        for node_id in frontier:
            nxt.update(graph.neighbors(node_id))
        frontier = nxt
        nodes_in_scope.update(nxt)
    return graph.subgraph(nodes_in_scope).copy()


def assign_rank_prizes(node_ids: list[str], query_emb: np.ndarray, node_emb_lookup: dict[str, np.ndarray], top_k: int) -> tuple[np.ndarray, np.ndarray]:
    embs = np.vstack([node_emb_lookup[node_id] for node_id in node_ids])
    similarities = cosine_similarity([query_emb], embs)[0]
    prizes = np.zeros(len(node_ids), dtype=float)
    ranked_idx = np.argsort(similarities)[::-1][: min(top_k, len(similarities))]
    for rank, idx in enumerate(ranked_idx):
        prizes[idx] = top_k - rank
    return prizes, similarities


def graph_to_pcst_input(graph: nx.Graph, prizes: np.ndarray, edge_cost: float = 1.0):
    node_list = list(graph.nodes())
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
    edge_pairs = []
    edge_costs = []
    for source, target in graph.edges():
        edge_pairs.append((node_to_idx[source], node_to_idx[target]))
        edge_costs.append(edge_cost)
    return (
        node_list,
        np.asarray(edge_pairs, dtype=np.int32),
        np.asarray(prizes, dtype=np.float64),
        np.asarray(edge_costs, dtype=np.float64),
    )


def run_node_prize_pcst(graph: nx.Graph, prizes: np.ndarray, edge_cost: float = 1.0, pruning: str = "strong"):
    node_list, edge_array, prize_array, cost_array = graph_to_pcst_input(graph, prizes, edge_cost=edge_cost)
    selected_vertex_idx, selected_edge_idx = pcst_fast(edge_array, prize_array, cost_array, -1, 1, pruning, 0)
    selected_nodes = [node_list[idx] for idx in selected_vertex_idx]
    selected_node_set = set(selected_nodes)
    selected_edges = []
    edge_list = list(graph.edges())
    for edge_idx in selected_edge_idx:
        source, target = edge_list[edge_idx]
        if source in selected_node_set and target in selected_node_set:
            selected_edges.append((source, target))
    return selected_nodes, selected_edges


def retrieve_unpruned_subgraph(
    query: str,
    query_encoder: SentenceTransformer,
    embedding_dim: int,
    full_graph: nx.Graph,
    node_embeddings: np.ndarray,
    node_ids: list[str],
    num_retrieved_seeds: int = NUM_RETRIEVED_SEEDS,
    k_hops: int = K_HOPS,
):
    query_emb = encode_query(query, query_encoder, embedding_dim=embedding_dim)
    seed_nodes, _ = retrieve_seed_nodes(query_emb, node_embeddings, node_ids=node_ids, k=num_retrieved_seeds)
    unpruned_graph = expand_k_hops(full_graph, seed_nodes, k=k_hops)

    retrieved_node_ids = list(unpruned_graph.nodes())
    retrieved_edges_df = pd.DataFrame(list(unpruned_graph.edges()), columns=["source", "target"])

    return {
        "query": query,
        "query_emb": query_emb,
        "seed_nodes": seed_nodes,
        "subgraph": unpruned_graph,
        "subgraph_node_ids": retrieved_node_ids,
        "subgraph_edges_df": retrieved_edges_df,
    }


def prune_retrieved_subgraph(
    retrieval_result,
    node_emb_lookup: dict[str, np.ndarray],
    prize_top_k: int = PRIZE_TOP_K,
    edge_cost: float = EDGE_COST,
):
    candidate_graph = retrieval_result["subgraph"]
    candidate_node_ids = retrieval_result["subgraph_node_ids"]

    candidate_node_emb_lookup = {
        node_id: node_emb_lookup[node_id]
        for node_id in candidate_node_ids
    }

    prizes, similarities = assign_rank_prizes(
        node_ids=candidate_node_ids,
        query_emb=retrieval_result["query_emb"],
        node_emb_lookup=candidate_node_emb_lookup,
        top_k=prize_top_k,
    )

    selected_nodes, selected_edges = run_node_prize_pcst(
        candidate_graph,
        prizes,
        edge_cost=edge_cost,
        pruning="strong",
    )

    node_scores_df = pd.DataFrame(
        {
            "node_id": candidate_node_ids,
            "similarity": similarities,
            "prize": prizes,
        }
    ).sort_values(["prize", "similarity"], ascending=False)
    selected_edges_df = pd.DataFrame(selected_edges, columns=["source", "target"])

    return {
        "query": retrieval_result["query"],
        "selected_nodes": selected_nodes,
        "selected_edges_df": selected_edges_df,
        "node_scores_df": node_scores_df,
    }


def generate_with_ollama(prompt: str, model: str = "qwen2.5:3b", timeout: int = 120) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        "http://localhost:11434/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except error.URLError as exc:
        raise RuntimeError(
            "Could not connect to Ollama at http://localhost:11434. "
            "Start Ollama with `ollama serve` and make sure `ollama pull qwen2.5:3b` is done."
        ) from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama returned non-JSON output.") from exc

    answer = str(parsed.get("response", "")).strip()
    if not answer:
        raise RuntimeError(f"Ollama response missing `response`: {parsed}")
    return answer


def main() -> None:
    query = input("Enter query: ").strip()
    if not query:
        raise SystemExit("Query cannot be empty.")

    config_path = resolve_path("configs/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    nodes = pd.read_csv(resolve_path(config["paths"]["nodes_csv"]))
    edges = pd.read_csv(resolve_path(config["paths"]["edges_csv"]))
    node_embedding_path = resolve_path(config["paths"]["node_features"])
    embedding_dim = int(config["embedding"]["embedding_dim"])
    node_embeddings = np.load(node_embedding_path)[:, :embedding_dim]

    nodes = nodes.copy()
    nodes["node_id"] = nodes["node_id"].astype(str)
    edges = edges.copy()
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)

    node_ids = nodes["node_id"].tolist()
    node_emb_lookup = dict(zip(node_ids, node_embeddings))
    full_graph = build_graph(edges, node_ids=node_ids)

    query_encoder = SentenceTransformer(config["embedding"]["model_name"])

    retrieval_result = retrieve_unpruned_subgraph(
        query=query,
        query_encoder=query_encoder,
        embedding_dim=embedding_dim,
        full_graph=full_graph,
        node_embeddings=node_embeddings,
        node_ids=node_ids,
        num_retrieved_seeds=NUM_RETRIEVED_SEEDS,
        k_hops=K_HOPS,
    )
    pruned_result = prune_retrieved_subgraph(
        retrieval_result,
        node_emb_lookup=node_emb_lookup,
        prize_top_k=PRIZE_TOP_K,
        edge_cost=EDGE_COST,
    )

    # Use post-prune nodes as evidence for prompt generation.
    post_table = get_text_rows(pruned_result["selected_nodes"], nodes)
    docs = post_table.to_dict(orient="records")
    prompt = build_prompt(query, docs)
    answer = generate_with_ollama(prompt, model="qwen2.5:3b")

    print(answer)


if __name__ == "__main__":
    main()
