import numpy as np
import pandas as pd
import torch
import networkx as nx
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Class for splitting data into train, validation, and test sets
    """

    def __init__(self, papers_df: pd.DataFrame, citation_graph: Optional[nx.Graph] = None,
                 node_mapping: Optional[Dict] = None):
        """
        Initialize the data splitter

        """
        self.papers_df = papers_df
        self.citation_graph = citation_graph
        self.node_mapping = node_mapping or {}

    def create_data_splits(self,
                           test_size: float = 0.2,
                           val_size: float = 0.1,
                           random_state: int = 42,
                           stratify_col: str = 'primary_category') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into train, validation, and test sets
        """
        logger.info(f"Creating data splits (test={test_size}, val={val_size}, stratify={stratify_col})...")

        # Filter out papers without a stratification category
        valid_papers = self.papers_df.dropna(subset=[stratify_col])

        # Filter out classes with too few samples
        min_samples_per_class = 5
        valid_papers = valid_papers.groupby(stratify_col).filter(lambda x: len(x) >= min_samples_per_class)

        # Log the number of papers and categories after filtering
        categories = valid_papers[stratify_col].unique()
        logger.info(f"Using {len(valid_papers)} papers across {len(categories)} categories for classification")

        # First, split into train+val and test
        train_val_indices, test_indices = train_test_split(
            np.arange(len(valid_papers)),
            test_size=test_size,
            random_state=random_state,
            stratify=valid_papers[stratify_col]
        )

        # Then split train+val into train and val
        val_ratio = val_size / (1 - test_size)
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_ratio,
            random_state=random_state,
            stratify=valid_papers.iloc[train_val_indices][stratify_col]
        )

        # Create DataFrames for each split
        train_df = valid_papers.iloc[train_indices].reset_index(drop=True)
        val_df = valid_papers.iloc[val_indices].reset_index(drop=True)
        test_df = valid_papers.iloc[test_indices].reset_index(drop=True)

        logger.info(
            f"Split dataset into train ({len(train_df)}), validation ({len(val_df)}), and test ({len(test_df)}) sets")

        return train_df, val_df, test_df

    def create_temporal_splits(self,
                               year_col: str = 'year',
                               train_end_year: int = 2017,
                               val_year: int = 2018,
                               test_start_year: int = 2019) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset temporally by publication year
        """
        logger.info(f"Creating temporal splits (train<={train_end_year}, val={val_year}, test>={test_start_year})...")

        # Convert year to numeric, handling non-numeric values
        self.papers_df[year_col] = pd.to_numeric(self.papers_df[year_col], errors='coerce')

        # Filter out papers without year information
        valid_papers = self.papers_df.dropna(subset=[year_col])

        # Create splits
        train_df = valid_papers[valid_papers[year_col] <= train_end_year].reset_index(drop=True)
        val_df = valid_papers[valid_papers[year_col] == val_year].reset_index(drop=True)
        test_df = valid_papers[valid_papers[year_col] >= test_start_year].reset_index(drop=True)

        logger.info(
            f"Split dataset into train ({len(train_df)}), validation ({len(val_df)}), and test ({len(test_df)}) sets")

        return train_df, val_df, test_df

    def create_data_splits_preserving_citations(self,
                                                test_size: float = 0.2,
                                                val_size: float = 0.1,
                                                random_state: int = 42,
                                                stratify_col: str = 'primary_category') -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the dataset while trying to preserve citation relationships
        """
        logger.info("Creating data splits while preserving citation structure...")

        # Create node masks for papers that are in your node mapping
        if self.citation_graph is None:
            logger.warning("Citation graph not provided, using regular splits")
            train_df, val_df, test_df = self.create_data_splits(
                test_size=test_size,
                val_size=val_size,
                random_state=random_state,
                stratify_col=stratify_col
            )
        else:
            # First do a stratified split on categories
            train_df, val_df, test_df = self.create_data_splits(
                test_size=test_size,
                val_size=val_size,
                random_state=random_state,
                stratify_col=stratify_col
            )

        # Create node masks for PyG data
        max_node_idx = max(self.node_mapping.values()) + 1

        train_mask = torch.zeros(max_node_idx, dtype=torch.bool)
        val_mask = torch.zeros(max_node_idx, dtype=torch.bool)
        test_mask = torch.zeros(max_node_idx, dtype=torch.bool)

        # Set masks based on split
        train_indices = [self.node_mapping[pid] for pid in train_df['paper_id'] if pid in self.node_mapping]
        val_indices = [self.node_mapping[pid] for pid in val_df['paper_id'] if pid in self.node_mapping]
        test_indices = [self.node_mapping[pid] for pid in test_df['paper_id'] if pid in self.node_mapping]

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        # Log statistics about the splits
        logger.info(f"Train set: {train_mask.sum().item()} nodes")
        logger.info(f"Validation set: {val_mask.sum().item()} nodes")
        logger.info(f"Test set: {test_mask.sum().item()} nodes")

        # Check citation relationships between splits
        if self.citation_graph is not None:
            train_nodes = set(np.where(train_mask.numpy())[0])
            val_nodes = set(np.where(val_mask.numpy())[0])
            test_nodes = set(np.where(test_mask.numpy())[0])

            # Count citations between sets
            train_to_val = sum(1 for u, v in self.citation_graph.edges() if u in train_nodes and v in val_nodes)
            train_to_test = sum(1 for u, v in self.citation_graph.edges() if u in train_nodes and v in test_nodes)
            val_to_test = sum(1 for u, v in self.citation_graph.edges() if u in val_nodes and v in test_nodes)

            logger.info(f"Citations from train to validation: {train_to_val}")
            logger.info(f"Citations from train to test: {train_to_test}")
            logger.info(f"Citations from validation to test: {val_to_test}")

        return train_mask, val_mask, test_mask