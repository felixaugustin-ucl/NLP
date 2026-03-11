from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
from typing import Any

import pandas as pd

from kg_io import ensure_source_tsv, merge_duplicate_nodes, read_tsv, resolve_source_text_path, write_table
from parse_refs import extract_explicit_references


ARTICLE_PAR_NODE_RE = re.compile(r"^(Article\s+\d+[A-Za-z]?)\((\d+)\)$")

ARTICLE_NODE_RE = re.compile(r"^Article\s+\d+[A-Za-z]?$")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def snippet_around(text: str, start: int, end: int, width: int = 120) -> str:
    left_room = max(0, (width - (end - start)) // 2)
    s = max(0, start - left_room)
    e = min(len(text), s + width)
    if e - s < width:
        s = max(0, e - width)
    return text[s:e].strip()


def ensure_annex_main_node(nodes_df: pd.DataFrame, annex_id: str) -> tuple[pd.DataFrame, bool]:
    if annex_id in set(nodes_df["node_id"]):
        return nodes_df, False
    prefix = annex_id + " "
    has_chunks = nodes_df["node_id"].astype(str).str.startswith(prefix).any()
    if not has_chunks:
        return nodes_df, False

    stub = pd.DataFrame(
        [{"node_id": annex_id, "kind": "ANNEX_STUB", "text": "", "seq": int(nodes_df["seq"].max()) + 1}]
    )
    nodes_df = pd.concat([nodes_df, stub], ignore_index=True)
    return nodes_df, True


def build_edges(nodes_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    node_ids = set(nodes_df["node_id"].astype(str))
    rows: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    counts = {"contains": 0, "refers_to": 0}

    # CONTAINS edges: Article X -> Article X(n)
    for row in nodes_df.itertuples(index=False):
        node_id = str(row.node_id)
        if str(row.kind) != "PARAGRAPH":
            continue
        m = ARTICLE_PAR_NODE_RE.match(node_id)
        if not m:
            continue
        parent = m.group(1)
        if parent not in node_ids:
            continue
        key = (parent, "CONTAINS", node_id)
        if key in seen_edges:
            continue
        rows.append(
            {
                "src": parent,
                "rel": "CONTAINS",
                "dst": node_id,
                "evidence_span": "",
                "evidence_text": "",
                "confidence": 1.0,
            }
        )
        seen_edges.add(key)
        counts["contains"] += 1

    # REFERS_TO edges: regex-only explicit references with evidence.
    for row in nodes_df.itertuples(index=False):
        src = str(row.node_id)
        src_text = str(row.text or "")
        if not src_text:
            continue

        for ref in extract_explicit_references(src_text):
            for dst in ref.dst_node_ids:
                if dst not in node_ids:
                    continue
                key = (src, "REFERS_TO", dst)
                if key in seen_edges:
                    continue
                rows.append(
                    {
                        "src": src,
                        "rel": "REFERS_TO",
                        "dst": dst,
                        "evidence_span": ref.evidence_span,
                        "evidence_text": snippet_around(src_text, ref.start, ref.end),
                        "confidence": 1.0,
                    }
                )
                seen_edges.add(key)
                counts["refers_to"] += 1

    edges_df = pd.DataFrame(
        rows,
        columns=["src", "rel", "dst", "evidence_span", "evidence_text", "confidence"],
    )
    return edges_df, counts, nodes_df


def _parent_article(node_id: str) -> str | None:
    if ARTICLE_NODE_RE.match(node_id):
        return node_id
    m = ARTICLE_PAR_NODE_RE.match(node_id)
    if m:
        return m.group(1)
    return None


def _top_counts_as_list(counter: Counter[str], n: int = 10) -> list[dict[str, Any]]:
    return [{"node_id": node_id, "count": int(count)} for node_id, count in counter.most_common(n)]


def build_summary(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> dict[str, Any]:
    edge_counts_by_rel = {k: int(v) for k, v in edges_df["rel"].value_counts().to_dict().items()}

    refers = edges_df[edges_df["rel"] == "REFERS_TO"].copy()
    refers["src_article"] = refers["src"].map(_parent_article)
    refers["dst_article"] = refers["dst"].map(lambda x: x if ARTICLE_NODE_RE.match(str(x)) else None)

    in_counter = Counter(
        refers.loc[refers["dst_article"].notna(), "dst_article"].astype(str).tolist()
    )

    article_level_pairs = (
        refers.loc[refers["src_article"].notna() & refers["dst_article"].notna(), ["src_article", "dst_article"]]
        .drop_duplicates()
    )
    out_counter = Counter(article_level_pairs["src_article"].astype(str).tolist())

    sample_edges = (
        edges_df.loc[edges_df["rel"] == "REFERS_TO"]
        .head(20)[["src", "rel", "dst", "evidence_span", "evidence_text", "confidence"]]
        .to_dict(orient="records")
    )

    return {
        "num_nodes": int(len(nodes_df)),
        "num_edges": int(len(edges_df)),
        "edge_counts_by_rel": edge_counts_by_rel,
        "top_10_articles_by_out_degree": _top_counts_as_list(out_counter, 10),
        "top_10_articles_by_in_degree": _top_counts_as_list(in_counter, 10),
        "sample_edges": sample_edges,
    }

def write_node_embeddings(nodes_df: pd.DataFrame, outdir: str | Path) -> str:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError(
            "Node embeddings require sentence-transformers with the all-MiniLM-L6-v2 model."
        ) from exc

    texts = nodes_df["text"].fillna("").astype(str).str.strip()
    texts = texts.where(texts.ne(""), nodes_df["node_id"].astype(str))
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(texts.tolist(), normalize_embeddings=True, show_progress_bar=False)
    embedding_values = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings

    path = Path(outdir) / "node_embeddings.parquet"
    embeddings_df = pd.DataFrame(
        {
            "node_id": nodes_df["node_id"].astype(str),
            "embedding": embedding_values,
        }
    )
    embeddings_df.to_parquet(path, index=False)
    return path.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Build auditable EU AI Act KG from explicit references.")
    parser.add_argument("--tsv", required=True, help="Input TSV path (seq, kind, locator, text).")
    parser.add_argument("--outdir", required=True, help="Output directory for nodes/edges/summary.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tsv_path, generated = ensure_source_tsv(args.tsv)
    if generated:
        source_text_path = resolve_source_text_path(tsv_path)
        print(f"Generated missing TSV from {source_text_path}: {tsv_path}")

    tsv_df = read_tsv(tsv_path)
    nodes_df, duplicate_count = merge_duplicate_nodes(tsv_df)

    # Add annex stub nodes only when annex chunk nodes exist but main annex node is missing.
    annex_targets = sorted(
        {
            f"ANNEX {m.group(1).upper()}"
            for text in nodes_df["text"].astype(str)
            for m in re.finditer(r"\bAnnex\s+([IVXLCDM]+)\b", text, flags=re.IGNORECASE)
        }
    )
    stubs_added = 0
    for annex_id in annex_targets:
        nodes_df, added = ensure_annex_main_node(nodes_df, annex_id)
        stubs_added += int(added)

    edges_df, edge_counts, nodes_df = build_edges(nodes_df)

    nodes_file = write_table(nodes_df, outdir, "nodes")
    edges_file = write_table(edges_df, outdir, "edges")
    edges_parquet_file = "edges.parquet"
    edges_df.to_parquet(outdir / edges_parquet_file, index=False)
    embeddings_file = write_node_embeddings(nodes_df, outdir)

    summary = build_summary(nodes_df, edges_df)
    summary_path = outdir / "graph_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Validation output requested by user.
    print(f"Duplicate locators merged: {duplicate_count}")
    print(f"Annex stub nodes added: {stubs_added}")
    print(f"Contains edges: {edge_counts['contains']}")
    print(f"Refers_to edges: {edge_counts['refers_to']}")
    print("Top 10 most cited articles (in-degree):")
    for item in summary["top_10_articles_by_in_degree"]:
        print(f"  {item['node_id']}: {item['count']}")
    print("Top 10 articles citing others most (article-level out-degree):")
    for item in summary["top_10_articles_by_out_degree"]:
        print(f"  {item['node_id']}: {item['count']}")
    print(f"Wrote {nodes_file}, {edges_file}, {edges_parquet_file}, {embeddings_file}, graph_summary.json to {outdir}")


if __name__ == "__main__":
    main()
