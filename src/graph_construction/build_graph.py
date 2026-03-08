"""
Build the PyTorch Geometric graph object from nodes and edges.

Converts nodes.csv + edges.csv + node_features.npy into a
PyG Data or HeteroData object.

Usage:
    python -m src.graph_construction.build_graph
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_node_id_mapping(nodes_df: pd.DataFrame) -> dict:
    """Create mapping from node_id string to integer index."""
    node_ids = nodes_df["node_id"].tolist()
    mapping = {nid: idx for idx, nid in enumerate(node_ids)}
    return mapping


def build_edge_index(
    edges_df: pd.DataFrame,
    node_id_mapping: dict
) -> tuple[torch.Tensor, list[str]]:
    """
    Convert edges DataFrame to PyG edge_index tensor.

    Returns:
        edge_index: [2, num_edges] tensor
        edge_types: list of edge type strings
    """
    sources = []
    targets = []
    edge_types = []

    for _, row in edges_df.iterrows():
        src = row["source"]
        tgt = row["target"]

        if src in node_id_mapping and tgt in node_id_mapping:
            sources.append(node_id_mapping[src])
            targets.append(node_id_mapping[tgt])
            edge_types.append(row.get("relation", "unknown"))

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    return edge_index, edge_types


def build_pyg_data(
    node_features: np.ndarray,
    edge_index: torch.Tensor,
    edge_types: list[str],
    node_id_mapping: dict,
) -> Data:
    """
    Build a PyTorch Geometric Data object.

    Args:
        node_features: numpy array of shape [num_nodes, feature_dim]
        edge_index: [2, num_edges] tensor
        edge_types: list of edge relation strings
        node_id_mapping: dict mapping node_id -> int index

    Returns:
        PyG Data object
    """
    x = torch.tensor(node_features, dtype=torch.float)

    # Encode edge types as integers
    unique_types = sorted(set(edge_types))
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    edge_type_tensor = torch.tensor(
        [type_to_idx[t] for t in edge_types], dtype=torch.long
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type_tensor,
        num_nodes=x.size(0),
    )

    # Store metadata
    data.node_id_mapping = node_id_mapping
    data.edge_type_mapping = type_to_idx

    logger.info(
        f"Built graph: {data.num_nodes} nodes, "
        f"{data.num_edges} edges, "
        f"{len(unique_types)} edge types"
    )
    return data


def main():
    config = load_config()

    # Load data
    nodes_df = pd.read_csv(config["paths"]["nodes_csv"])
    edges_df = pd.read_csv(config["paths"]["edges_csv"])
    node_features = np.load(config["paths"]["node_features"])

    # Build mapping
    node_id_mapping = build_node_id_mapping(nodes_df)

    # Save mapping
    mapping_path = config["paths"]["node_id_mapping"]
    Path(mapping_path).parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_path, "w") as f:
        json.dump(node_id_mapping, f, indent=2)

    # Build edge index
    edge_index, edge_types = build_edge_index(edges_df, node_id_mapping)

    # Build PyG data
    data = build_pyg_data(node_features, edge_index, edge_types, node_id_mapping)

    # Save
    graph_path = config["paths"]["graph_object"]
    Path(graph_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, graph_path)
    logger.info(f"Saved graph to: {graph_path}")


if __name__ == "__main__":
    main()
