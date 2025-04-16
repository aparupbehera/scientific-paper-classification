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
                 combination: str = 'cross_attention'):
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

        # Feature projection layers
        self.text_projector = nn.Linear(text_dim, hidden_dim)
        self.graph_projector = nn.Linear(graph_dim, hidden_dim)

        # Cross-attention mechanism
        if self.combination == 'cross_attention':
            # Multi-head attention for text features attending to graph features
            self.query_proj = nn.Linear(hidden_dim, hidden_dim)
            self.key_proj = nn.Linear(hidden_dim, hidden_dim)
            self.value_proj = nn.Linear(hidden_dim, hidden_dim)
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                batch_first=True,
                dropout=dropout / 2
            )
            # Project concatenated attention output and text features
            self.fusion_layer = nn.Linear(2 * hidden_dim, hidden_dim)

        # Weighted feature fusion
        elif self.combination == 'weighted':
            # Learnable weights for text and graph features
            self.text_weight = nn.Parameter(torch.tensor(0.6))
            self.graph_weight = nn.Parameter(torch.tensor(0.4))

        # Gating mechanism
        elif self.combination == 'gate':
            # Gate that learns to balance text and graph information
            self.gate_network = nn.Sequential(
                nn.Linear(text_dim + graph_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            )

        # Create GNN layers
        self.convs = nn.ModuleList()

        # Input layer GNN
        if self.gnn_type == 'gcn':
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        elif self.gnn_type == 'gat':
            # GAT with 4 attention heads, each with hidden_dim/4 features
            self.convs.append(GATConv(hidden_dim, hidden_dim // 4, heads=4))
        elif self.gnn_type == 'sage':
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        # Layer normalization after each GNN layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Hidden layers GNN
        for _ in range(num_layers - 2):
            if self.gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif self.gnn_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim // 4, heads=4))
            elif self.gnn_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        # Output layer GNN
        if self.gnn_type == 'gcn':
            self.convs.append(GCNConv(hidden_dim, output_dim))
        elif self.gnn_type == 'gat':
            self.convs.append(GATConv(hidden_dim, output_dim, heads=1))
        elif self.gnn_type == 'sage':
            self.convs.append(SAGEConv(hidden_dim, output_dim))

        # Skip connection linear projections (for dimension matching)
        self.skip_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])

        # Final classification head
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # Split input into text and graph parts
        text_x = x[:, :self.text_dim]
        graph_x = x[:, self.text_dim:]

        # Project both to same dimension space
        text_proj = self.text_projector(text_x)
        graph_proj = self.graph_projector(graph_x)

        # Apply the combination method
        if self.combination == 'cross_attention':
            # Prepare for multi-head attention (batch_size, seq_len, hidden_dim)
            query = self.query_proj(text_proj).unsqueeze(1)  # [N, 1, H]
            key = self.key_proj(graph_proj).unsqueeze(1)  # [N, 1, H]
            value = self.value_proj(graph_proj).unsqueeze(1)  # [N, 1, H]

            # Apply cross-attention
            attn_output, _ = self.attention(query, key, value)
            attn_output = attn_output.squeeze(1)  # [N, H]

            # Combine attention output with original text features
            combined = torch.cat([text_proj, attn_output], dim=1)
            x = self.fusion_layer(combined)

        elif self.combination == 'weighted':
            # Normalize weights with softmax
            weights = F.softmax(torch.stack([self.text_weight, self.graph_weight]), dim=0)
            x = weights[0] * text_proj + weights[1] * graph_proj

        elif self.combination == 'gate':
            # Calculate gate values
            gate_input = torch.cat([text_x, graph_x], dim=1)
            gate_values = self.gate_network(gate_input)

            # Apply gating mechanism
            x = gate_values * text_proj + (1 - gate_values) * graph_proj

        else:  # Default to concatenation
            x = torch.cat([text_proj, graph_proj], dim=1)
            x = nn.Linear(text_proj.size(1) + graph_proj.size(1), self.hidden_dim).to(x.device)(x)

        # Apply GNN layers with layer norm, ReLU, and skip connections
        for i in range(self.num_layers - 1):
            # Store input for skip connection
            identity = x

            # Apply GNN layer
            x = self.convs[i](x, edge_index)

            # Apply layer normalization
            x = self.layer_norms[i](x)

            # Apply ReLU activation
            x = F.relu(x)

            # Apply skip connection
            x = x + self.skip_projections[i](identity)

            # Apply dropout
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer
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