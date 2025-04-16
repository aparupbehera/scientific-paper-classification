"""
Configuration file for scientific paper classification
"""

# Data configuration
DATA_DIR = './data'
METADATA_FILE = 'arxiv-metadata-oai-snapshot.json'
CITATION_FILE = 'arxiv-citations.txt'
CATEGORIES = None  # List of categories or None for all
MAX_PAPERS = 5000  # Number of papers to process

USE_OGB_CITATIONS = False  # Backup citation graph, use local for now
OGB_DATASET = 'ogbn-arxiv'  # OGB dataset name

# Output configuration
OUTPUT_DIR = './output'
PROCESSED_DIR = './data/processed'
EMBEDDINGS_DIR = './data/processed/embeddings'
FIGURES_DIR = './output/figures'
LOGS_DIR = './output/logs'

# Embedding configuration
TEXT_MODEL = 'scibert'  # Options: 'scibert', 'tfidf'
GRAPH_MODEL = 'node2vec'  # Options: 'node2vec', 'deepwalk'
EMBEDDING_DIM = 128
SCIBERT_POOLING = 'mean'  # Options: 'cls', 'mean', 'max'
USE_CACHE = False  # Whether to use cached embeddings

# Split configuration
TEST_SIZE = 0.2
VAL_SIZE = 0.1
TEMPORAL_SPLIT = False
TRAIN_END_YEAR = 2017

# Training configuration
MODELS_TO_TRAIN = ['text', 'graph', 'hybrid']
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.5
GNN_TYPE = 'gat'  # Options: 'gcn', 'gat', 'sage'
COMBINATION = 'cross_attention'  # Options: 'cross_attention', 'weighted', 'gate'
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
PATIENCE = 10

# Misc configuration
SEED = 42
VERBOSE = False
VISUALIZE = True