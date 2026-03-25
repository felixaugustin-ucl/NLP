"""
GNN Model Definitions.

Implements:
- GraphSAGE (primary)
- GAT (alternative)
- R-GCN (extension for typed edges)

All models take node features and graph structure as input
and produce graph-aware node embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    RGCNConv,
    SAGEConv,
)


class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE encoder for learning graph-aware node embeddings.

    Architecture:
        Input features → SAGEConv → ReLU → Dropout → SAGEConv → L2 Normalize

    The learned embeddings capture both textual content and
    structural/relational context in the legal graph.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 768,
        out_channels: int = 768,
        num_layers: int = 2,
        dropout: float = 0.3,
        aggr: str = "mean",
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.norms.append(nn.BatchNorm1d(hidden_channels))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.norms.append(nn.BatchNorm1d(hidden_channels))

        # Final layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: node features [num_nodes, in_channels]
            edge_index: edge index [2, num_edges]

        Returns:
            Node embeddings [num_nodes, out_channels], L2 normalized
        """
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer (no activation, no dropout)
        x = self.convs[-1](x, edge_index)

        # L2 normalize for cosine similarity retrieval
        x = F.normalize(x, p=2, dim=-1)

        return x


class GATEncoder(nn.Module):
    """
    Graph Attention Network encoder.

    Uses attention to weight neighbor importance,
    useful when some neighbors are noisy.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 768,
        out_channels: int = 768,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        concat: bool = True,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, concat=concat)
        )
        first_out = hidden_channels * heads if concat else hidden_channels
        self.norms.append(nn.BatchNorm1d(first_out))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(first_out, hidden_channels, heads=heads, concat=concat)
            )
            self.norms.append(nn.BatchNorm1d(first_out))

        # Final layer (single head, no concat)
        if num_layers > 1:
            self.convs.append(
                GATConv(first_out, out_channels, heads=1, concat=False)
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x


class RGCNEncoder(nn.Module):
    """
    Relational Graph Convolutional Network encoder.

    Handles multiple edge types (part_of, refers_to, defines, etc.)
    with separate weight matrices per relation.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 768,
        out_channels: int = 768,
        num_relations: int = 5,
        num_layers: int = 2,
        num_bases: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(
            RGCNConv(in_channels, hidden_channels, num_relations, num_bases=num_bases)
        )
        self.norms.append(nn.BatchNorm1d(hidden_channels))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(
                RGCNConv(
                    hidden_channels, hidden_channels, num_relations,
                    num_bases=num_bases
                )
            )
            self.norms.append(nn.BatchNorm1d(hidden_channels))

        # Final layer
        if num_layers > 1:
            self.convs.append(
                RGCNConv(
                    hidden_channels, out_channels, num_relations,
                    num_bases=num_bases
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: node features [num_nodes, in_channels]
            edge_index: [2, num_edges]
            edge_type: edge type indices [num_edges]
        """
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_type)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index, edge_type)
        x = F.normalize(x, p=2, dim=-1)
        return x


def get_model(config: dict, in_channels: int) -> nn.Module:
    """
    Factory function to create the appropriate GNN model.

    Args:
        config: GNN section of config.yaml
        in_channels: input feature dimension

    Returns:
        GNN encoder module
    """
    arch = config["architecture"]
    gnn_config = config

    if arch == "GraphSAGE":
        return GraphSAGEEncoder(
            in_channels=in_channels,
            hidden_channels=gnn_config["hidden_channels"],
            out_channels=gnn_config["out_channels"],
            num_layers=gnn_config["num_layers"],
            dropout=gnn_config["dropout"],
            aggr=gnn_config.get("aggregation", "mean"),
        )
    elif arch == "GAT":
        gat_cfg = gnn_config.get("gat", {})
        return GATEncoder(
            in_channels=in_channels,
            hidden_channels=gnn_config["hidden_channels"],
            out_channels=gnn_config["out_channels"],
            num_layers=gnn_config["num_layers"],
            heads=gat_cfg.get("heads", 4),
            dropout=gnn_config["dropout"],
            concat=gat_cfg.get("concat", True),
        )
    elif arch == "RGCN":
        rgcn_cfg = gnn_config.get("rgcn", {})
        return RGCNEncoder(
            in_channels=in_channels,
            hidden_channels=gnn_config["hidden_channels"],
            out_channels=gnn_config["out_channels"],
            num_relations=rgcn_cfg.get("num_relations", 5),
            num_layers=gnn_config["num_layers"],
            num_bases=rgcn_cfg.get("num_bases", 4),
            dropout=gnn_config["dropout"],
        )
    else:
        raise ValueError(f"Unknown GNN architecture: {arch}")
