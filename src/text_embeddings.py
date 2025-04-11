import os
import torch
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union, Callable

logger = logging.getLogger(__name__)


class TextEmbeddings:
    """
    Class for generating text embeddings from paper content
    """

    def __init__(self, papers_df: pd.DataFrame, cache_dir: str = './embeddings'):
        """
        Initialize the text embeddings generator
        """
        self.papers_df = papers_df
        self.cache_dir = cache_dir

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

    def prepare_scibert_embeddings(self,
                                   model_name: str = "allenai/scibert_scivocab_uncased",
                                   batch_size: int = 16,
                                   pooling: str = "mean",
                                   max_length: int = 512,
                                   text_col: str = 'text',
                                   use_cache: bool = True,
                                   cache_file: str = "scibert_embeddings.pt") -> torch.Tensor:
        """
        Generate text embeddings using SciBERT with different pooling strategies
        """
        cache_path = os.path.join(self.cache_dir, cache_file)

        # Check if cached embeddings exist
        if use_cache and os.path.exists(cache_path):
            logger.info(f"Loading cached SciBERT embeddings from {cache_path}")
            try:
                text_embeddings = torch.load(cache_path)
                logger.info(f"Loaded {text_embeddings.shape[0]} embeddings of dimension {text_embeddings.shape[1]}")
                return text_embeddings
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}. Generating new embeddings.")

        logger.info(f"Preparing text embeddings using {model_name} with {pooling} pooling...")

        try:
            # Import transformers dynamically
            from transformers import AutoTokenizer, AutoModel

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)

            # Move model to GPU if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {device}")
            model.to(device)
            model.eval()

            # Generate embeddings in batches
            embeddings = []
            for i in tqdm(range(0, len(self.papers_df), batch_size)):
                batch_texts = self.papers_df[text_col].iloc[i:i + batch_size].tolist()

                # Tokenize
                inputs = tokenizer(batch_texts, padding=True, truncation=True,
                                   return_tensors="pt", max_length=max_length)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Generate embeddings
                with torch.no_grad():
                    outputs = model(**inputs)

                # Apply the selected pooling strategy
                if pooling == "cls":
                    # CLS token as the embedding (first token)
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
                elif pooling == "mean":
                    # Mean pooling - average all tokens
                    attention_mask = inputs['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                        input_mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = batch_embeddings.cpu()
                elif pooling == "max":
                    # Max pooling
                    attention_mask = inputs['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
                    batch_embeddings = torch.max(token_embeddings, 1)[0].cpu()
                else:
                    raise ValueError(f"Unknown pooling strategy: {pooling}")

                embeddings.append(batch_embeddings)

            # Concatenate all embeddings
            all_embeddings = torch.cat(embeddings, dim=0)
            logger.info(f"Generated {all_embeddings.shape[0]} embeddings of dimension {all_embeddings.shape[1]}")

            # Cache embeddings
            if use_cache:
                logger.info(f"Caching SciBERT embeddings to {cache_path}")
                torch.save(all_embeddings, cache_path)

            return all_embeddings

        except ImportError:
            logger.error("Transformers not installed. Install it with: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"Error generating SciBERT embeddings: {e}")
            raise

    def prepare_tf_idf_embeddings(self,
                                  text_col: str = 'text',
                                  max_features: int = 5000,
                                  use_cache: bool = True,
                                  cache_file: str = "tfidf_embeddings.pt") -> torch.Tensor:
        """
        Generate text embeddings using TF-IDF
        """
        cache_path = os.path.join(self.cache_dir, cache_file)

        # Check if cached embeddings exist
        if use_cache and os.path.exists(cache_path):
            logger.info(f"Loading cached TF-IDF embeddings from {cache_path}")
            try:
                text_embeddings = torch.load(cache_path)
                logger.info(f"Loaded {text_embeddings.shape[0]} embeddings of dimension {text_embeddings.shape[1]}")
                return text_embeddings
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}. Generating new embeddings.")

        logger.info(f"Generating TF-IDF embeddings (max_features={max_features})...")

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            # Create vectorizer
            vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')

            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(self.papers_df[text_col])

            # Convert to dense tensor
            text_embeddings = torch.tensor(tfidf_matrix.todense(), dtype=torch.float)

            logger.info(
                f"Generated {text_embeddings.shape[0]} TF-IDF embeddings of dimension {text_embeddings.shape[1]}")

            # Cache embeddings
            if use_cache:
                logger.info(f"Caching TF-IDF embeddings to {cache_path}")
                torch.save(text_embeddings, cache_path)

            return text_embeddings

        except ImportError:
            logger.error("Scikit-learn not installed. Install it with: pip install scikit-learn")
            raise
        except Exception as e:
            logger.error(f"Error generating TF-IDF embeddings: {e}")
            raise