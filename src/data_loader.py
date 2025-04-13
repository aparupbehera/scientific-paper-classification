import os
import json
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import logging
import config
from typing import List, Dict, Tuple, Optional, Union

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Class for loading and preprocessing scientific paper datasets
    """
    def __init__(self, data_dir: str, categories: Optional[List[str]] = None, max_papers: Optional[int] = None):
        self.data_dir = data_dir
        self.categories = categories
        self.max_papers = max_papers
        self.papers = None
        self.citation_graph = None
        self.node_mapping = {}  # Maps paper IDs to node indices

        # Creating output directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

    def load_arxiv_dataset(self, metadata_file: str) -> pd.DataFrame:
        logger.info(f"Loading ArXiv dataset from {metadata_file}...")
        papers = []

        try:
            # Loading metadata
            with open(os.path.join(self.data_dir, metadata_file), 'r') as f:
                for i, line in tqdm(enumerate(f)):
                    if self.max_papers and i >= self.max_papers:
                        break

                    paper = json.loads(line)

                    # Filtering by categories if specified
                    if self.categories:
                        paper_categories = paper.get('categories', '').split()
                        if not any(cat in paper_categories for cat in self.categories):
                            continue

                    # Extract relevant fields
                    paper_data = {
                        'paper_id': paper.get('id'),
                        'title': paper.get('title', ''),
                        'abstract': paper.get('abstract', ''),
                        'categories': paper.get('categories', ''),
                        'authors': paper.get('authors_parsed', []),
                        'year': paper.get('journal-ref', '').split()[-1] if paper.get('journal-ref') else None,
                    }

                    # Skipping papers without abstract
                    if not paper_data['abstract']:
                        continue

                    papers.append(paper_data)

            # Converts to DataFrame
            self.papers = pd.DataFrame(papers)

            # Creates mapping from original IDs to consecutive indices
            for idx, paper_id in enumerate(self.papers['paper_id']):
                self.node_mapping[paper_id] = idx

            logger.info(f"Loaded {len(self.papers)} papers")

            # Basic data cleaning
            self._clean_data()

            return self.papers

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def _clean_data(self):
        """Cleaning and normalizing the dataset"""

        # Handle missing values
        self.papers['abstract'] = self.papers['abstract'].fillna('')
        self.papers['title'] = self.papers['title'].fillna('')

        # Truncating overly long abstracts (for efficiency)
        max_abstract_length = 50000
        self.papers['abstract'] = self.papers['abstract'].apply(
            lambda x: x[:max_abstract_length] if len(x) > max_abstract_length else x
        )

        # Combining title and abstract for better embeddings
        self.papers['text'] = self.papers.apply(
            lambda row: f"{row['title']} {row['abstract']}",
            axis=1
        )

        logger.info("Data cleaning completed")

    def load_citations(self, citation_file=None):
        """
        Loading citation links from file or OGB dataset
        """
        logger.info("Loading citation graph...")
        self.citation_graph = nx.DiGraph()

        # Add all papers as nodes
        for idx in range(len(self.papers)):
            self.citation_graph.add_node(idx)

        if config.USE_OGB_CITATIONS:
            try:
                from ogb.nodeproppred import PygNodePropPredDataset

                # Load OGB dataset
                logger.info(f"Loading OGB citation network from {config.OGB_DATASET}...")
                dataset = PygNodePropPredDataset(name=config.OGB_DATASET)
                data = dataset[0]

                # Extract OGB edge index (citation network)
                ogb_edge_index = data.edge_index.numpy()

                # OGB dataset has node_idx -> paper properties
                # But we need to map OGB indices to our papers

                # First try to get ArXiv IDs from OGB data
                # (This implementation depends on how OGB stores ArXiv IDs)

                # APPROACH 1: Try to match by paper title
                # Create lookup from title to our index
                title_to_idx = {}
                for idx, title in enumerate(self.papers['title']):
                    if isinstance(title, str):
                        # Normalize title for matching
                        norm_title = title.lower().strip()
                        title_to_idx[norm_title] = idx

                # Create mappings for edges
                edges = []
                edge_count = 0

                # Add edges from OGB (will need customization based on OGB format)
                for i in range(ogb_edge_index.shape[1]):
                    src = ogb_edge_index[0, i]
                    dst = ogb_edge_index[1, i]

                    # Map to your indices if possible
                    # This is placeholder logic - you'll need to adjust based on actual data
                    # For now, we'll just add edges that are between papers in our dataset
                    if src < len(self.papers) and dst < len(self.papers):
                        edges.append((src, dst))
                        edge_count += 1

                # Add all edges to graph
                self.citation_graph.add_edges_from(edges)
                logger.info(f"Added {edge_count} citation edges from OGB dataset")

            except ImportError:
                logger.error("Failed to import OGB. Install with 'pip install ogb'")
                raise
            except Exception as e:
                logger.error(f"Error loading OGB citations: {e}")
                raise
        else:
            # Original citation loading code
            try:
                # Add citation links
                edges = []
                with open(os.path.join(self.data_dir, citation_file), 'r') as f:
                    for line in tqdm(f):
                        source, target = line.strip().split()
                        if source in self.node_mapping and target in self.node_mapping:
                            edges.append((self.node_mapping[source], self.node_mapping[target]))

                self.citation_graph.add_edges_from(edges)

            except Exception as e:
                logger.error(f"Error loading citations: {e}")
                raise

        logger.info(f"Loaded citation graph with {self.citation_graph.number_of_nodes()} nodes and {self.citation_graph.number_of_edges()} edges")
        return self.citation_graph

    def extract_primary_categories(self) -> pd.Series:
        """
        Extracting primary categories from papers for classification
        """
        if self.papers is None:
            logger.error("Papers not loaded yet")
            raise ValueError("Papers not loaded yet")

        # Extracting primary category (first category in the list)
        # Each paper can have multiple categories which may result in duplicates
        # Hence extracting the primary category as main label
        self.papers['primary_category'] = self.papers['categories'].apply(
            lambda x: x.split()[0] if isinstance(x, str) and x else None
        )

        # Count category distribution
        category_counts = self.papers['primary_category'].value_counts()
        logger.info("\nCategory Distribution (top 10):")
        for cat, count in category_counts.head(10).items():
            logger.info(f"{cat}: {count} papers")

        # Filter categories with too few samples
        min_samples = 3
        valid_categories = category_counts[category_counts >= min_samples].index
        self.papers['valid_category'] = self.papers['primary_category'].apply(
            lambda x: x if x in valid_categories else None
        )

        logger.info(f"Retained {len(valid_categories)} categories with at least {min_samples} papers each")

        return category_counts