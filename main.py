import os
import torch
import logging
import config

# Import from src modules
from src.utils import setup_logging, set_seed, create_directory_structure, get_available_device
from src.data_loader import DataLoader
from src.graph_embeddings import GraphEmbeddings
from src.text_embeddings import TextEmbeddings
from src.data_splitter import DataSplitter
from src.visualization import Visualization
from src.converters import DataConverters

def main():
    # Create directory structure
    create_directory_structure(config.OUTPUT_DIR, dirs=[
        'data', 'embeddings', 'figures', 'logs', 'models'
    ])

    # Make sure processed directory exists
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)

    # Set up logging
    logger = setup_logging(config.LOGS_DIR,
                          log_level=logging.DEBUG if config.VERBOSE else logging.INFO)

    # Set random seed
    set_seed(config.SEED)

    logger.info("Starting Scientific Paper Classification")
    logger.info(f"Using data from {config.DATA_DIR}")

    # Load custom dataset
    data_loader = DataLoader(
        data_dir=config.DATA_DIR,
        categories=config.CATEGORIES,
        max_papers=config.MAX_PAPERS
    )

    # Validate that input files exist
    metadata_path = os.path.join(config.DATA_DIR, config.METADATA_FILE)
    citation_path = os.path.join(config.DATA_DIR, config.CITATION_FILE)

    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found at {metadata_path}")
        return
    if not os.path.exists(citation_path):
        logger.error(f"Citation file not found at {citation_path}")
        return

    # Load papers and citations
    logger.info(f"Loading ArXiv metadata from {config.METADATA_FILE}...")
    papers_df = data_loader.load_arxiv_dataset(config.METADATA_FILE)

    logger.info("Extracting primary categories...")
    category_counts = data_loader.extract_primary_categories()

    if config.USE_OGB_CITATIONS:
        logger.info("Using OGB citation network...")
        citation_graph = data_loader.load_citations()
    else:
        logger.info(f"Loading citations from {config.CITATION_FILE}...")
        citation_graph = data_loader.load_citations(config.CITATION_FILE)

    # Print basic statistics
    logger.info(f"Loaded {len(papers_df)} papers across {len(category_counts)} categories")
    logger.info(
        f"Loaded citation graph with {citation_graph.number_of_nodes()} nodes and {citation_graph.number_of_edges()} edges")

    # Generate embeddings
    # Text embeddings
    logger.info("Generating text embeddings...")
    text_embedder = TextEmbeddings(
        papers_df=papers_df,
        cache_dir=config.EMBEDDINGS_DIR
    )

    if config.TEXT_MODEL == 'scibert':
        text_embeddings = text_embedder.prepare_scibert_embeddings(
            pooling=config.SCIBERT_POOLING,
            use_cache=config.USE_CACHE
        )
    elif config.TEXT_MODEL == 'tfidf':
        text_embeddings = text_embedder.prepare_tf_idf_embeddings(
            use_cache=config.USE_CACHE
        )
    else:
        logger.error(f"Unknown text embedding model: {config.TEXT_MODEL}")
        return

    # Graph embeddings
    logger.info("Generating graph embeddings...")
    graph_embedder = GraphEmbeddings(
        graph=citation_graph,
        embedding_dim=config.EMBEDDING_DIM,
        cache_dir=config.EMBEDDINGS_DIR
    )

    if config.GRAPH_MODEL == 'node2vec':
        graph_embeddings = graph_embedder.generate_node2vec_embeddings(
            dimensions=config.EMBEDDING_DIM,
            use_cache=config.USE_CACHE
        )
    elif config.GRAPH_MODEL == 'deepwalk':
        graph_embeddings = graph_embedder.generate_deep_walk_embeddings(
            dimensions=config.EMBEDDING_DIM,
            use_cache=config.USE_CACHE
        )
    else:
        logger.error(f"Unknown graph embedding model: {config.GRAPH_MODEL}")
        return

    # Split data
    logger.info("Creating data splits...")
    splitter = DataSplitter(
        papers_df=papers_df,
        citation_graph=citation_graph,
        node_mapping=data_loader.node_mapping
    )

    if config.TEMPORAL_SPLIT:
        train_df, val_df, test_df = splitter.create_temporal_splits(
            train_end_year=config.TRAIN_END_YEAR
        )
    else:
        train_df, val_df, test_df = splitter.create_data_splits(
            test_size=config.TEST_SIZE,
            val_size=config.VAL_SIZE
        )

    # Create PyTorch Geometric data objects
    logger.info("Creating data masks...")
    train_mask, val_mask, test_mask = splitter.create_data_splits_preserving_citations(
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE
    )

    # Create node labels
    logger.info("Creating node labels...")
    categories = papers_df['primary_category'].dropna().unique()
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}

    node_labels = torch.full((len(papers_df),), -1, dtype=torch.long)
    for idx, cat in enumerate(papers_df['primary_category']):
        if isinstance(cat, str) and cat in category_to_idx:
            node_labels[idx] = category_to_idx[cat]

    # Create converters
    logger.info("Converting to PyTorch Geometric format...")
    converters = DataConverters(papers_df, citation_graph)

    if train_mask.shape[0] != text_embeddings.shape[0]:
        logger.warning(
            f"Mask shape ({train_mask.shape[0]}) doesn't match embeddings shape ({text_embeddings.shape[0]})")
        # Truncate to smaller size
        min_size = min(train_mask.shape[0], text_embeddings.shape[0])
        train_mask = train_mask[:min_size]
        val_mask = val_mask[:min_size]
        test_mask = test_mask[:min_size]
        text_embeddings = text_embeddings[:min_size]
        graph_embeddings = graph_embeddings[:min_size]
        logger.warning(f"Truncated to {min_size} samples")

    # Convert to PyG format
    text_data = converters.convert_to_pytorch_geometric(
        node_features=text_embeddings,
        node_labels=node_labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    graph_data = converters.convert_to_pytorch_geometric(
        node_features=graph_embeddings,
        node_labels=node_labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    hybrid_data = converters.convert_to_hybrid_pyg_data(
        text_embeddings=text_embeddings,
        graph_embeddings=graph_embeddings,
        node_labels=node_labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    # Create data save directory
    data_save_dir = config.PROCESSED_DIR
    os.makedirs(data_save_dir, exist_ok=True)

    # Save PyG data
    logger.info(f"Saving PyG data to {data_save_dir}...")
    torch.save(text_data, os.path.join(data_save_dir, 'text_data.pt'))
    torch.save(graph_data, os.path.join(data_save_dir, 'graph_data.pt'))
    torch.save(hybrid_data, os.path.join(data_save_dir, 'hybrid_data.pt'))

    # Also save category mapping for later reference
    torch.save(category_to_idx, os.path.join(data_save_dir, 'category_mapping.pt'))

    logger.info(f"Saved PyG data to {data_save_dir}")

    # Generate visualizations if requested
    if config.VISUALIZE:
        logger.info("Generating visualizations...")
        visualizer = Visualization(os.path.join(config.OUTPUT_DIR, 'figures'))

        # Visualize category distribution
        visualizer.plot_category_distribution(
            category_counts=category_counts,
            title='Paper Category Distribution',
            filename='category_distribution.png'
        )

        # Visualize citation distribution
        visualizer.plot_citation_distribution(
            graph=citation_graph,
            title='Citation Distribution',
            filename='citation_distribution.png'
        )

        # Visualize embeddings
        valid_indices = node_labels != -1
        try:
            visualizer.plot_embeddings_tsne(
                embeddings=text_embeddings[valid_indices],
                labels=node_labels[valid_indices],
                title='Text Embeddings t-SNE',
                filename='text_embeddings_tsne.png'
            )

            visualizer.plot_embeddings_tsne(
                embeddings=graph_embeddings[valid_indices],
                labels=node_labels[valid_indices],
                title='Graph Embeddings t-SNE',
                filename='graph_embeddings_tsne.png'
            )
        except Exception as e:
            logger.error(f"Error generating embeddings visualization: {e}")

        logger.info("Visualizations completed")

    logger.info("Processing completed successfully")


if __name__ == '__main__':
    main()