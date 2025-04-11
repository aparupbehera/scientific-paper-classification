import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MLPClassifier(nn.Module):
    """
    Simple MLP classifier for text-only or graph-only embeddings
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2, dropout: float = 0.5):
        """
        Initialize the MLP classifier
        """
        super(MLPClassifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Create layers
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer
        if num_layers > 1:
            self.layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            # If only one layer, connect input directly to output
            self.layers[0] = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        # Apply layers with ReLU activation and dropout
        for i in range(self.num_layers - 1):
            x = self.layers[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply final layer without activation or dropout
        if self.num_layers > 0:
            x = self.layers[-1](x)

        return x


class GNNClassifier(nn.Module):
    """
    GNN classifier for graph-based classification
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.5, gnn_type: str = 'gcn'):
        """
        Initialize the GNN classifier
        """
        super(GNNClassifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type.lower()

        # Create GNN layers
        self.convs = nn.ModuleList()

        # Input layer
        if self.gnn_type == 'gcn':
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif self.gnn_type == 'gat':
            self.convs.append(GATConv(input_dim, hidden_dim))
        elif self.gnn_type == 'sage':
            self.convs.append(SAGEConv(input_dim, hidden_dim))
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

        # Hidden layers
        for _ in range(num_layers - 2):
            if self.gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif self.gnn_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim))
            elif self.gnn_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        # Output layer
        if num_layers > 1:
            if self.gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, output_dim))
            elif self.gnn_type == 'gat':
                self.convs.append(GATConv(hidden_dim, output_dim))
            elif self.gnn_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        # Apply GNN layers with ReLU activation and dropout
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply final layer without activation or dropout
        if self.num_layers > 0:
            x = self.convs[-1](x, edge_index)

        return x


class HybridGNN(nn.Module):
    """
    Hybrid GNN model combining text and graph information
    """

    def __init__(self, text_dim: int, graph_dim: int = 0, hidden_dim: int = 256, output_dim: int = 0,
                 num_layers: int = 2, dropout: float = 0.5, gnn_type: str = 'gcn',
                 combination: str = 'concatenate'):
        """
        Initialize the hybrid GNN model
        """
        super(HybridGNN, self).__init__()

        self.text_dim = text_dim
        self.graph_dim = graph_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type.lower()
        self.combination = combination.lower()

        # Input dimension depends on combination method
        if self.combination == 'concatenate':
            self.input_dim = text_dim + graph_dim
        elif self.combination in ['add', 'gate']:
            # For add and gate, project both to same dimension first
            self.input_dim = hidden_dim
            self.text_projector = nn.Linear(text_dim, hidden_dim)
            self.graph_projector = nn.Linear(graph_dim, hidden_dim)

            if self.combination == 'gate':
                # Create gating mechanism
                self.gate = nn.Sequential(
                    nn.Linear(text_dim + graph_dim, hidden_dim),
                    nn.Sigmoid()
                )
        else:
            raise ValueError(f"Unknown combination method: {combination}")

        # Create GNN layers
        self.convs = nn.ModuleList()

        # Input layer
        if self.gnn_type == 'gcn':
            self.convs.append(GCNConv(self.input_dim, hidden_dim))
        elif self.gnn_type == 'gat':
            self.convs.append(GATConv(self.input_dim, hidden_dim))
        elif self.gnn_type == 'sage':
            self.convs.append(SAGEConv(self.input_dim, hidden_dim))
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

        # Hidden layers
        for _ in range(num_layers - 2):
            if self.gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif self.gnn_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim))
            elif self.gnn_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        # Output layer
        if num_layers > 1:
            if self.gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, output_dim))
            elif self.gnn_type == 'gat':
                self.convs.append(GATConv(hidden_dim, output_dim))
            elif self.gnn_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        # If using separate text and graph embeddings
        if self.combination != 'concatenate':
            # Split the input into text and graph parts
            text_x = x[:, :self.text_dim]
            graph_x = x[:, self.text_dim:]

            # Project them to the same dimension
            text_proj = self.text_projector(text_x)
            graph_proj = self.graph_projector(graph_x)

            if self.combination == 'add':
                # Simple addition
                x = text_proj + graph_proj
            elif self.combination == 'gate':
                # Gated combination
                gate_input = torch.cat([text_x, graph_x], dim=1)
                gate_value = self.gate(gate_input)
                x = gate_value * text_proj + (1 - gate_value) * graph_proj

        # Apply GNN layers with ReLU activation and dropout
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply final layer without activation or dropout
        if self.num_layers > 0:
            x = self.convs[-1](x, edge_index)

        return x


class TextOnlyBaseline(nn.Module):
    """
    Text-only baseline model
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2, dropout: float = 0.5):
        """
        Initialize the text-only baseline model
        """
        super(TextOnlyBaseline, self).__init__()

        self.classifier = MLPClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass
        """
        return self.classifier(x)


class GraphOnlyBaseline(nn.Module):
    """
    Graph-only baseline model
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.5):
        """
        Initialize the graph-only baseline model
        """
        super(GraphOnlyBaseline, self).__init__()

        self.classifier = MLPClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass
        """
        return self.classifier(x)