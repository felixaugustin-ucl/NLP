"""
Node Feature Generator.

Generates numeric feature vectors for each graph node:
- Text embeddings (sentence-transformers)
- Optional metadata features (node type, article number, text length, etc.)

Output: data/embeddings/node_features.npy

Usage:
    python -m src.features.generate_features
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_text_embeddings(
    texts: list[str],
    model_name: str | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Generate dense text embeddings for each node.

    Args:
        texts: list of node text strings
        model_name: HuggingFace model identifier
        batch_size: encoding batch size

    Returns:
        numpy array of shape [num_nodes, embedding_dim]
    """
    if not model_name:
        raise ValueError("model_name must be provided. Use config['embedding']['model_name'].")

    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info(f"Encoding {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings


def generate_metadata_features(nodes_df: pd.DataFrame) -> np.ndarray:
    """
    Generate optional metadata features for each node.

    Features:
    - Node type one-hot encoding
    - Text length (normalized)
    - Article number (normalized)
    - Is definition (binary)
    - Is annex (binary)
    - Is recital (binary)

    Returns:
        numpy array of shape [num_nodes, num_metadata_features]
    """
    features = []

    # Node type one-hot
    if "type" in nodes_df.columns:
        type_dummies = pd.get_dummies(nodes_df["type"], prefix="type")
        features.append(type_dummies.values)

    # Text length (normalized)
    if "text" in nodes_df.columns:
        text_lengths = nodes_df["text"].str.len().fillna(0).values
        text_lengths = text_lengths / (text_lengths.max() + 1e-8)
        features.append(text_lengths.reshape(-1, 1))

    if features:
        metadata = np.hstack(features).astype(np.float32)
        logger.info(f"Metadata features shape: {metadata.shape}")
        return metadata

    return np.zeros((len(nodes_df), 0), dtype=np.float32)


def combine_features(
    text_embeddings: np.ndarray,
    metadata_features: np.ndarray,
) -> np.ndarray:
    """
    Concatenate text embeddings with metadata features.

    Final feature vector per node: [text_embedding ; metadata_features]
    """
    if metadata_features.shape[1] == 0:
        return text_embeddings

    combined = np.hstack([text_embeddings, metadata_features])
    logger.info(f"Combined feature shape: {combined.shape}")
    return combined


def main():
    config = load_config()

    # Load nodes
    nodes_df = pd.read_csv(config["paths"]["nodes_csv"])
    texts = nodes_df["text"].fillna("").tolist()

    # Generate embeddings
    emb_config = config["embedding"]
    text_embeddings = generate_text_embeddings(
        texts,
        model_name=emb_config["model_name"],
        batch_size=emb_config["batch_size"],
    )

    # Generate metadata features
    metadata_features = generate_metadata_features(nodes_df)

    # Combine
    node_features = combine_features(text_embeddings, metadata_features)

    # Save
    output_path = config["paths"]["node_features"]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, node_features)
    logger.info(f"Saved node features to: {output_path}")

    # Save node ID mapping
    mapping = {nid: idx for idx, nid in enumerate(nodes_df["node_id"])}
    mapping_path = config["paths"]["node_id_mapping"]
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    logger.info(f"Saved node ID mapping to: {mapping_path}")


if __name__ == "__main__":
    main()
