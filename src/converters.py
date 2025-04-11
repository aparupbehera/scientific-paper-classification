import torch
import numpy as np
import pandas as pd
import networkx as nx
import logging
from typing import List, Dict, Tuple, Optional, Union
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class DataConverters:
    """
    Class for converting data to different formats
    """

    def __init__(self, papers_df: Optional[pd.DataFrame] = None, citation_graph: Optional[nx.Graph] = None):
        self.papers_df = papers_df
        self.citation_graph = citation_graph

    def convert_to_pytorch_geometric(self,
                                     node_features: torch.Tensor,
                                     edge_index: Optional[torch.Tensor] = None,
                                     node_labels: Optional[torch.Tensor] = None,
                                     train_mask: Optional[torch.Tensor] = None,
                                     val_mask: Optional[torch.Tensor] = None,
                                     test_mask: Optional[torch.Tensor] = None) -> Data:

        #Convert data to PyTorch Geometric format
        logger.info("Converting to PyTorch Geometric format...")

        # Create edge index if not provided
        if edge_index is None and self.citation_graph is not None:
            edge_index = torch.tensor(list(self.citation_graph.edges())).t().contiguous()

        # Create labels if not provided
        if node_labels is None and self.papers_df is not None and 'primary_category' in self.papers_df.columns:
            # Convert categories to numeric labels
            categories = self.papers_df['primary_category'].dropna().unique()
            category_to_idx = {cat: idx for idx, cat in enumerate(categories)}

            # Create labels tensor
            node_labels = torch.full((len(self.papers_df),), -1, dtype=torch.long)

            for idx, cat in enumerate(self.papers_df['primary_category']):
                if pd.notna(cat) and cat in category_to_idx:
                    node_labels[idx] = category_to_idx[cat]

        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=node_labels
        )

        # Add masks if provided
        if train_mask is not None:
            data.train_mask = train_mask
        if val_mask is not None:
            data.val_mask = val_mask
        if test_mask is not None:
            data.test_mask = test_mask

        # Add metadata
        if node_labels is not None:
            data.num_classes = int(node_labels.max()) + 1

        logger.info(f"Created PyTorch Geometric Data with {data.num_nodes} nodes and {data.num_edges} edges")
        return data

    def convert_to_hybrid_pyg_data(self,
                                   text_embeddings: torch.Tensor,
                                   graph_embeddings: torch.Tensor,
                                   edge_index: Optional[torch.Tensor] = None,
                                   node_labels: Optional[torch.Tensor] = None,
                                   train_mask: Optional[torch.Tensor] = None,
                                   val_mask: Optional[torch.Tensor] = None,
                                   test_mask: Optional[torch.Tensor] = None) -> Data:

        #Convert to PyG format with both text and graph embeddings for the hybrid model
        logger.info("Converting to hybrid PyTorch Geometric format...")

        # Check for embedding size mismatch
        if text_embeddings.shape[0] != graph_embeddings.shape[0]:
            logger.error(
                f"Mismatch in embedding shapes: text={text_embeddings.shape[0]}, graph={graph_embeddings.shape[0]}")
            raise ValueError("Text and graph embeddings have different numbers of nodes")

        # Concatenate embeddings
        hybrid_x = torch.cat([text_embeddings, graph_embeddings], dim=1)
        logger.info(f"Created hybrid embeddings with shape {hybrid_x.shape}")

        # Create PyG data with hybrid embeddings
        data = self.convert_to_pytorch_geometric(
            node_features=hybrid_x,
            edge_index=edge_index,
            node_labels=node_labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )

        # Add additional metadata
        data.text_dim = text_embeddings.shape[1]
        data.graph_dim = graph_embeddings.shape[1]

        logger.info(f"Created hybrid PyTorch Geometric Data: {data}")
        return data

    @staticmethod
    def networkx_to_edge_index(graph: nx.Graph) -> torch.Tensor:
        """
        Convert a NetworkX graph to PyTorch Geometric edge_index
        """
        return torch.tensor(list(graph.edges())).t().contiguous()