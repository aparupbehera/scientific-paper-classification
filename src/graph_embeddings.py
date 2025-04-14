import os
import torch
import numpy as np
import networkx as nx
import logging
from tqdm import tqdm
from typing import Optional

logger = logging.getLogger(__name__)


class GraphEmbeddings:
    """
    Class for generating embeddings from graph structure
    """

    def __init__(self, graph: nx.Graph, embedding_dim: int = 256, cache_dir: str = './embeddings'):
        """
        Initialize the graph embeddings generator
        """
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.cache_dir = cache_dir

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

    def generate_node2vec_embeddings(self,
                                     dimensions: int = 256,
                                     walk_length: int = 100,
                                     num_walks: int = 20,
                                     workers: int = 4,
                                     p: float = 0.25,
                                     q: float = 0.25,
                                     use_cache: bool = True,
                                     cache_file: str = "node2vec_embeddings.pt") -> torch.Tensor:
        """
        Generate Node2Vec embeddings from the citation graph
        """
        cache_path = os.path.join(self.cache_dir, cache_file)

        # Check if cached embeddings exist
        if use_cache and os.path.exists(cache_path):
            logger.info(f"Loading cached Node2Vec embeddings from {cache_path}")
            try:
                graph_embeddings = torch.load(cache_path)
                logger.info(f"Loaded {graph_embeddings.shape[0]} embeddings of dimension {graph_embeddings.shape[1]}")
                return graph_embeddings
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}. Generating new embeddings.")

        logger.info(f"Generating Node2Vec embeddings (d={dimensions}, walks={num_walks}, length={walk_length})...")

        try:
            # Import Node2Vec dynamically
            from node2vec import Node2Vec

            # Initialize Node2Vec
            node2vec = Node2Vec(
                self.graph,
                dimensions=dimensions,
                walk_length=walk_length,
                num_walks=num_walks,
                workers=workers,
                p=p,
                q=q
            )

            # Train model
            logger.info("Training Node2Vec model...")
            model = node2vec.fit(window=10, min_count=1)

            # Extract embeddings
            logger.info("Extracting node embeddings...")
            node_embeddings = []
            for node in tqdm(range(self.graph.number_of_nodes())):
                if str(node) in model.wv:
                    emb = model.wv[str(node)]
                else:
                    # Use zeros for nodes not in any walk
                    emb = np.zeros(dimensions)
                node_embeddings.append(emb)

            # Convert to tensor
            graph_embeddings = torch.tensor(np.array(node_embeddings), dtype=torch.float)
            logger.info(f"Generated {len(node_embeddings)} Node2Vec embeddings of dimension {dimensions}")

            # Cache embeddings
            if use_cache:
                logger.info(f"Caching Node2Vec embeddings to {cache_path}")
                torch.save(graph_embeddings, cache_path)

            return graph_embeddings

        except ImportError:
            logger.error("node2vec package not installed. Install it with: pip install node2vec")
            raise
        except Exception as e:
            logger.error(f"Error generating Node2Vec embeddings: {e}")
            raise

    def generate_deep_walk_embeddings(self,
                                      dimensions: int = 256,
                                      walk_length: int = 100,
                                      num_walks: int = 20,
                                      workers: int = 4,
                                      use_cache: bool = True,
                                      cache_file: str = "deepwalk_embeddings.pt") -> torch.Tensor:
        """
        Generate DeepWalk embeddings (special case of Node2Vec with p=1, q=1)
        """
        return self.generate_node2vec_embeddings(
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=workers,
            p=0.25,
            q=0.25,
            use_cache=use_cache,
            cache_file=cache_file
        )