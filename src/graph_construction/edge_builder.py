"""
Edge Builder for the EU AI Act Graph.

Constructs relationships between legal nodes:
- Structural edges (part_of, contains)
- Cross-reference edges (refers_to)
- Definition edges (defines, uses_term)
- Optional: semantic similarity edges (similar_to)

Output: data/processed/edges.csv

Usage:
    python -m src.graph_construction.edge_builder
"""

import logging
import re
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Roman numeral mapping for annexes
ROMAN_TO_INT = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
    "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
}


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_structural_edges(nodes_df: pd.DataFrame) -> list[dict]:
    """
    Build structural (hierarchical) edges using parent_id.

    - Article → Paragraph  (contains / part_of)
    - Article 3 → Definition  (contains / part_of)
    - Annex → Annex item  (contains / part_of)
    """
    edges = []
    valid_ids = set(nodes_df["node_id"].tolist())

    for _, row in nodes_df.iterrows():
        parent_id = row.get("parent_id")
        if pd.notna(parent_id) and parent_id in valid_ids:
            child_id = row["node_id"]

            # Parent contains child
            edges.append({
                "source": parent_id,
                "target": child_id,
                "relation": "contains",
            })

            # Child is part of parent
            edges.append({
                "source": child_id,
                "target": parent_id,
                "relation": "part_of",
            })

    logger.info(f"Built {len(edges)} structural edges")
    return edges


def build_cross_reference_edges(nodes_df: pd.DataFrame) -> list[dict]:
    """
    Build cross-reference edges by detecting explicit references in text.

    Patterns matched:
    - "Article X" → refers_to article_X
    - "Article X(Y)" or "Article X, paragraph Y" → refers_to article_X_para_Y
    - "Annex III" → refers_to annex_III
    """
    edges = []
    valid_ids = set(nodes_df["node_id"].tolist())

    # Pre-compile patterns
    article_ref_pattern = re.compile(
        r'Article\s+(\d+)'
        r'(?:\s*\((\d+)\)|\s*,\s*paragraph\s+(\d+))?'
    )
    annex_ref_pattern = re.compile(r'Annex\s+([IVX]+)')

    for _, row in nodes_df.iterrows():
        source_id = row["node_id"]
        text = str(row.get("text", ""))
        source_art_num = row.get("article_number")

        # Find article references
        for match in article_ref_pattern.finditer(text):
            ref_art = int(match.group(1))
            ref_para = match.group(2) or match.group(3)

            # Build target node_id
            if ref_para:
                target_id = f"article_{ref_art}_para_{int(ref_para)}"
            else:
                target_id = f"article_{ref_art}"

            # Don't create self-references
            if target_id == source_id:
                continue

            # Don't create reference from paragraph to its own article
            if source_id.startswith(f"article_{ref_art}_para_"):
                if target_id == f"article_{ref_art}":
                    continue

            # Don't create reference from article to its own paragraphs
            if source_id == f"article_{ref_art}" and target_id.startswith(f"article_{ref_art}_para_"):
                continue

            if target_id in valid_ids:
                edges.append({
                    "source": source_id,
                    "target": target_id,
                    "relation": "refers_to",
                })

        # Find annex references
        for match in annex_ref_pattern.finditer(text):
            annex_num = match.group(1)
            target_id = f"annex_{annex_num}"

            if target_id == source_id:
                continue

            if target_id in valid_ids:
                edges.append({
                    "source": source_id,
                    "target": target_id,
                    "relation": "refers_to",
                })

    # Deduplicate edges
    seen = set()
    unique_edges = []
    for edge in edges:
        key = (edge["source"], edge["target"], edge["relation"])
        if key not in seen:
            seen.add(key)
            unique_edges.append(edge)

    logger.info(f"Built {len(unique_edges)} cross-reference edges (deduplicated from {len(edges)})")
    return unique_edges


def build_definition_edges(nodes_df: pd.DataFrame) -> list[dict]:
    """
    Build definition usage edges.

    For each definition node, find other nodes that use the defined term.
    Creates:
    - definition → article/paragraph: "defines" (the definition defines a concept)
    - article/paragraph → definition: "uses_term" (the node uses the defined term)
    """
    edges = []

    # Get definition nodes
    def_nodes = nodes_df[nodes_df["type"] == "definition"]
    other_nodes = nodes_df[nodes_df["type"].isin(["article", "paragraph", "recital"])]

    if def_nodes.empty:
        logger.info("No definition nodes found — skipping definition edges")
        return edges

    # Extract the term from each definition title (format: "Definition: term_name")
    def_terms = {}
    for _, row in def_nodes.iterrows():
        title = row["title"]
        if title.startswith("Definition: "):
            term = title[len("Definition: "):]
            # Only match terms with 3+ characters to avoid noise
            if len(term) >= 3:
                def_terms[row["node_id"]] = term.lower()

    logger.info(f"Matching {len(def_terms)} definition terms against {len(other_nodes)} nodes")

    # Search for term usage in other nodes
    for _, node_row in other_nodes.iterrows():
        node_id = node_row["node_id"]
        node_text = str(node_row.get("text", "")).lower()

        # Don't create edges from Article 3 itself to its own definitions
        if node_id == "article_3":
            continue

        for def_id, term in def_terms.items():
            # Check if the term appears in this node's text
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, node_text):
                edges.append({
                    "source": node_id,
                    "target": def_id,
                    "relation": "uses_term",
                })

    # Deduplicate
    seen = set()
    unique_edges = []
    for edge in edges:
        key = (edge["source"], edge["target"], edge["relation"])
        if key not in seen:
            seen.add(key)
            unique_edges.append(edge)

    logger.info(f"Built {len(unique_edges)} definition usage edges")
    return unique_edges


def build_all_edges(nodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main edge construction pipeline.
    """
    all_edges = []

    all_edges.extend(build_structural_edges(nodes_df))
    all_edges.extend(build_cross_reference_edges(nodes_df))
    all_edges.extend(build_definition_edges(nodes_df))

    df = pd.DataFrame(all_edges)
    logger.info(f"Total edges constructed: {len(df)}")

    # Summary statistics
    if not df.empty:
        counts = df["relation"].value_counts()
        for relation, count in counts.items():
            logger.info(f"  {relation}: {count}")

    return df


def save_edges(df: pd.DataFrame, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved edges to: {output_path}")


def main():
    config = load_config()
    nodes_df = pd.read_csv(config["paths"]["nodes_csv"])

    edges_df = build_all_edges(nodes_df)
    save_edges(edges_df, config["paths"]["edges_csv"])


if __name__ == "__main__":
    main()
