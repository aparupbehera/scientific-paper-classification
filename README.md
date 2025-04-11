# Scientific Paper Classification using Hybrid Graph and Text Embeddings

This project implements a hybrid approach to scientific paper classification that combines textual embeddings from SciBERT with graph-based embeddings from citation networks.

## Project Overview

With the rapid growth of scientific research, classifying papers into relevant sub-fields has become increasingly important. Traditional text-based classification often fails to capture structural relationships among papers, particularly citation networks. This project proposes a hybrid approach integrating textual embeddings from SciBERT and graph-based embeddings from Node2Vec and Graph Neural Networks (GNNs).

### Key Features

- Text-based classification using SciBERT embeddings
- Graph-based classification using Node2Vec embeddings of citation networks
- Hybrid GNN model that combines both text and graph information
- Comparative analysis of all three approaches
- Visualization of embeddings and classification results

## Installation

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Data Setup

This project uses ArXiv metadata and citation information. To set up the data:

1. Download the ArXiv dataset from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)
2. Place the `arxiv-metadata-oai-snapshot.json` file in the `data/` directory

## Running the Project

### Step 1: Data Processing

Process the data and generate embeddings:

```bash
python main.py
```

**Note**: This step may take significant time depending on the number of papers and your hardware. The processed data will be saved to `./data/processed/`.

### Step 2: Model Training

Train the selected models:

```bash
python train.py
```

Results will be saved to `./output/`.