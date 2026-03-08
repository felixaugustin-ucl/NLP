"""
Self-Supervised Training for the GNN.

Implements contrastive learning with negative sampling to train
graph-aware node embeddings without manual relevance labels.

Training objective:
- Positive pairs: connected nodes (refers_to, part_of, defines, etc.)
- Negative pairs: random unconnected nodes
- Loss: InfoNCE / contrastive loss

Usage:
    python -m src.gnn.train
"""

import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.gnn.models import get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------------------------
# Edge splitting
# -------------------------------------------------------------------------

def split_edges(
    edge_index: torch.Tensor,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict:
    """
    Split edges into train / val / test sets.

    Args:
        edge_index: [2, num_edges]
        train_ratio: fraction for training
        val_ratio: fraction for validation

    Returns:
        dict with 'train', 'val', 'test' edge_index tensors
    """
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges)

    train_end = int(num_edges * train_ratio)
    val_end = int(num_edges * (train_ratio + val_ratio))

    return {
        "train": edge_index[:, perm[:train_end]],
        "val": edge_index[:, perm[train_end:val_end]],
        "test": edge_index[:, perm[val_end:]],
    }


# -------------------------------------------------------------------------
# Sampling
# -------------------------------------------------------------------------

def sample_contrastive_pairs(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_negatives: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample anchor, positive, and negative nodes for contrastive learning.

    Positive pairs come from edges in the graph.
    Negative pairs are randomly sampled non-neighbors.

    Returns:
        anchors: [num_edges]
        positives: [num_edges]
        negatives: [num_edges, num_negatives]
    """
    anchors = edge_index[0]
    positives = edge_index[1]

    # Sample negatives
    num_pairs = anchors.size(0)
    negatives = torch.randint(0, num_nodes, (num_pairs, num_negatives))

    return anchors, positives, negatives


# -------------------------------------------------------------------------
# Loss functions
# -------------------------------------------------------------------------

def infonce_loss(
    anchor_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    negative_embs: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss.

    Encourages high similarity between anchor-positive pairs
    and low similarity between anchor-negative pairs.

    Args:
        anchor_emb: [batch_size, dim]
        positive_emb: [batch_size, dim]
        negative_embs: [batch_size, num_negatives, dim]
        temperature: temperature scaling factor

    Returns:
        Scalar loss
    """
    # Positive similarity
    pos_sim = F.cosine_similarity(anchor_emb, positive_emb, dim=-1)  # [batch]
    pos_sim = pos_sim / temperature

    # Negative similarity
    anchor_expanded = anchor_emb.unsqueeze(1)  # [batch, 1, dim]
    neg_sim = F.cosine_similarity(
        anchor_expanded, negative_embs, dim=-1
    )  # [batch, num_neg]
    neg_sim = neg_sim / temperature

    # InfoNCE: log(exp(pos) / (exp(pos) + sum(exp(neg))))
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch, 1 + num_neg]
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

    loss = F.cross_entropy(logits, labels)
    return loss


# -------------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------------

def train_epoch(
    model: torch.nn.Module,
    data,
    train_edges: torch.Tensor,
    optimizer,
    config: dict,
    device: torch.device,
) -> float:
    """Run one training epoch."""
    model.train()

    # Forward pass: compute all node embeddings
    x = data.x.to(device)
    edge_index = train_edges.to(device)

    # Get edge_type if using R-GCN
    edge_type = None
    if hasattr(data, "edge_type") and config["gnn"]["architecture"] == "RGCN":
        # Only use training edge types
        edge_type = data.edge_type[:train_edges.size(1)].to(device)
        embeddings = model(x, edge_index, edge_type)
    else:
        embeddings = model(x, edge_index)

    # Sample contrastive pairs
    anchors, positives, negatives = sample_contrastive_pairs(
        train_edges,
        num_nodes=data.num_nodes,
        num_negatives=config["training"]["num_negatives"],
    )

    # Get embeddings for pairs
    anchor_emb = embeddings[anchors.to(device)]
    positive_emb = embeddings[positives.to(device)]
    negative_emb = embeddings[negatives.to(device)]

    # Compute loss
    loss = infonce_loss(
        anchor_emb,
        positive_emb,
        negative_emb,
        temperature=config["training"]["temperature"],
    )

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data,
    val_edges: torch.Tensor,
    train_edges: torch.Tensor,
    config: dict,
    device: torch.device,
) -> float:
    """Evaluate on validation edges."""
    model.eval()

    x = data.x.to(device)
    edge_index = train_edges.to(device)

    if hasattr(data, "edge_type") and config["gnn"]["architecture"] == "RGCN":
        edge_type = data.edge_type[:train_edges.size(1)].to(device)
        embeddings = model(x, edge_index, edge_type)
    else:
        embeddings = model(x, edge_index)

    # Evaluate with contrastive loss on val edges
    anchors, positives, negatives = sample_contrastive_pairs(
        val_edges,
        num_nodes=data.num_nodes,
        num_negatives=config["training"]["num_negatives"],
    )

    anchor_emb = embeddings[anchors.to(device)]
    positive_emb = embeddings[positives.to(device)]
    negative_emb = embeddings[negatives.to(device)]

    loss = infonce_loss(
        anchor_emb,
        positive_emb,
        negative_emb,
        temperature=config["training"]["temperature"],
    )

    return loss.item()


def train(config: dict):
    """Main training function."""
    set_seed(config["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load graph
    data = torch.load(config["paths"]["graph_object"])
    logger.info(f"Loaded graph: {data.num_nodes} nodes, {data.num_edges} edges")

    # Split edges
    edge_splits = split_edges(
        data.edge_index,
        train_ratio=config["training"]["train_ratio"],
        val_ratio=config["training"]["val_ratio"],
    )

    # Create model
    in_channels = data.x.size(1)
    model = get_model(config["gnn"], in_channels).to(device)
    logger.info(f"Model: {config['gnn']['architecture']}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Scheduler
    scheduler = None
    if config["training"].get("scheduler") == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=config["training"]["epochs"]
        )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["training"]["patience"]

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss = train_epoch(
            model, data, edge_splits["train"], optimizer, config, device
        )

        val_loss = evaluate(
            model, data, edge_splits["val"], edge_splits["train"], config, device
        )

        if scheduler:
            scheduler.step()

        # Logging
        if epoch % config["logging"]["log_every"] == 0:
            logger.info(
                f"Epoch {epoch:04d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            ckpt_path = config["paths"]["model_checkpoint"]
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Periodic checkpoint
        if epoch % config["logging"]["save_every"] == 0:
            ckpt = f"models/checkpoints/graphsage_epoch{epoch}.pt"
            Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt)

    # ---- Save final embeddings ----
    logger.info("Generating final node embeddings...")
    model.load_state_dict(torch.load(config["paths"]["model_checkpoint"]))
    model.eval()

    with torch.no_grad():
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)

        if hasattr(data, "edge_type") and config["gnn"]["architecture"] == "RGCN":
            edge_type = data.edge_type.to(device)
            final_embeddings = model(x, edge_index, edge_type)
        else:
            final_embeddings = model(x, edge_index)

    emb_np = final_embeddings.cpu().numpy()
    emb_path = config["paths"]["final_embeddings"]
    Path(emb_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, emb_np)
    logger.info(f"Saved final embeddings ({emb_np.shape}) to: {emb_path}")


def main():
    config = load_config()
    train(config)


if __name__ == "__main__":
    main()
