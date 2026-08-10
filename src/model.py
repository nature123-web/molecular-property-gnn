"""Message-passing neural network for molecular property prediction.

Implemented directly in PyTorch rather than with PyTorch Geometric, so the
batching scheme and the aggregation are visible.

Molecules in a batch are combined into one **disconnected** graph, with node
indices offset per molecule and a ``batch`` vector recording which molecule each
node belongs to. This is how every GNN library batches variable-sized graphs: it
avoids padding to the largest molecule (which for a batch containing one big
molecule wastes most of the compute) and lets a single scatter-add do the
readout for the whole batch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .chem import ATOM_FEATURE_DIM, BOND_FEATURE_DIM


def scatter_sum(source: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Sum ``source`` rows into ``n`` buckets given by ``index``."""
    shape = (n,) + source.shape[1:]
    out = torch.zeros(shape, dtype=source.dtype, device=source.device)
    return out.index_add_(0, index, source)


def scatter_mean(source: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    total = scatter_sum(source, index, n)
    counts = scatter_sum(torch.ones(len(index), 1, device=source.device),
                         index, n)
    return total / counts.clamp(min=1)


def scatter_max(source: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    shape = (n,) + source.shape[1:]
    out = torch.full(shape, float("-inf"), dtype=source.dtype,
                     device=source.device)
    out = out.index_reduce_(0, index, source, reduce="amax", include_self=True)
    # Isolated nodes (a single-atom molecule) receive no messages; -inf there
    # would propagate nan through the head.
    return torch.where(torch.isinf(out), torch.zeros_like(out), out)


class MessagePassingLayer(nn.Module):
    """One round of edge-conditioned message passing with a GRU update.

    The message from j to i is a function of both endpoint states *and* the bond
    features, so a double bond and a single bond between the same atom types
    produce different messages -- which matters, since bond order is one of the
    strongest predictors of reactivity and physical properties.

    The GRU update (rather than a plain MLP on the concatenation) is what keeps
    deep stacks stable: it gives each node a gated choice about how much of the
    incoming message to absorb, which prevents the over-smoothing that makes all
    node states converge after a few layers.
    """

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.0
                 ) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_features: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            # A molecule with no bonds still has to come out the other side.
            return h
        source, target = edge_index[0], edge_index[1]
        messages = self.message_mlp(
            torch.cat([h[source], h[target], edge_features], dim=-1)
        )
        aggregated = scatter_sum(messages, target, h.shape[0])
        return self.update(self.dropout(aggregated), h)


class MolecularGNN(nn.Module):
    """Message-passing network with a set2set-style readout."""

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_tasks: int = 1,
        dropout: float = 0.1,
        readout: str = "mean_max",
        node_dim: int = ATOM_FEATURE_DIM,
        edge_dim: int = BOND_FEATURE_DIM,
    ) -> None:
        super().__init__()
        self.readout = readout
        self.embed = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, edge_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])

        readout_dim = hidden_dim * (2 if readout == "mean_max" else 1)
        self.head = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_tasks),
        )

    def encode(self, nodes: torch.Tensor, edge_index: torch.Tensor,
               edge_features: torch.Tensor) -> torch.Tensor:
        h = self.embed(nodes)
        for layer, norm in zip(self.layers, self.norms):
            h = norm(layer(h, edge_index, edge_features))
        return h

    def pool(self, h: torch.Tensor, batch: torch.Tensor, n_graphs: int
             ) -> torch.Tensor:
        if self.readout == "sum":
            # Sum is the right choice for extensive properties (anything that
            # scales with molecule size, like molecular weight); mean is right
            # for intensive ones. Getting this backwards costs a lot of accuracy.
            return scatter_sum(h, batch, n_graphs)
        if self.readout == "mean":
            return scatter_mean(h, batch, n_graphs)
        if self.readout == "max":
            return scatter_max(h, batch, n_graphs)
        if self.readout == "mean_max":
            return torch.cat([scatter_mean(h, batch, n_graphs),
                              scatter_max(h, batch, n_graphs)], dim=-1)
        raise ValueError(f"unknown readout '{self.readout}'")

    def forward(self, batch_data: dict) -> torch.Tensor:
        h = self.encode(batch_data["nodes"], batch_data["edge_index"],
                        batch_data["edge_features"])
        pooled = self.pool(h, batch_data["batch"], batch_data["n_graphs"])
        return self.head(pooled)

    def atom_embeddings(self, batch_data: dict) -> torch.Tensor:
        """Final per-atom states, for inspecting what the model learned."""
        return self.encode(batch_data["nodes"], batch_data["edge_index"],
                           batch_data["edge_features"])


def collate_graphs(graphs: list[dict]) -> dict:
    """Combine molecules into one disconnected graph.

    Node indices in each molecule's ``edge_index`` are offset by the running
    node count so edges never cross between molecules. Forgetting the offset is
    the classic bug here: it silently wires molecules together and the model
    still trains, just badly.
    """
    node_blocks, edge_blocks, edge_feature_blocks, batch_index = [], [], [], []
    offset = 0
    for graph_index, graph in enumerate(graphs):
        n = graph["n_atoms"]
        node_blocks.append(torch.as_tensor(graph["nodes"]))
        edge_blocks.append(torch.as_tensor(graph["edge_index"]) + offset)
        edge_feature_blocks.append(torch.as_tensor(graph["edge_features"]))
        batch_index.append(torch.full((n,), graph_index, dtype=torch.long))
        offset += n

    return {
        "nodes": torch.cat(node_blocks).float(),
        "edge_index": torch.cat(edge_blocks, dim=1).long(),
        "edge_features": torch.cat(edge_feature_blocks).float(),
        "batch": torch.cat(batch_index),
        "n_graphs": len(graphs),
        "n_nodes": offset,
    }
