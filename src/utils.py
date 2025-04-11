import os
import logging
import torch
import numpy as np
import random
from typing import List, Optional


def setup_logging(log_dir: str = './logs', log_level: int = logging.INFO) -> logging.Logger:

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, 'preprocessing.log'))
    file_handler.setLevel(log_level)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_available_device() -> torch.device:
    """
    GPU or CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def create_directory_structure(base_dir: str = './', dirs: Optional[List[str]] = None):
    if dirs is None:
        dirs = ['data', 'models', 'logs', 'figures', 'embeddings']

    for dir_name in dirs:
        os.makedirs(os.path.join(base_dir, dir_name), exist_ok=True)