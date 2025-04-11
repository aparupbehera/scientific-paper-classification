import os
import time
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for training and evaluating models
    """

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 criterion: Callable, device: torch.device, model_name: str = 'model',
                 output_dir: str = './output', patience: int = 10, scheduler=None):
        """
        Initialize the trainer
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model_name = model_name
        self.output_dir = output_dir
        self.patience = patience
        self.scheduler = scheduler

        # Create model directory if it doesn't exist
        self.model_dir = os.path.join(output_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)

        # Initialize history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rate': []
        }

        # Initialize counters
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

    def train_epoch(self, data: Data) -> Tuple[float, float, float]:
        """
        Train for one epoch
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Move data to device
        x = data.x.to(self.device)
        y = data.y.to(self.device)
        edge_index = data.edge_index.to(self.device) if hasattr(data, 'edge_index') else None
        train_mask = data.train_mask.to(self.device)

        # Forward pass
        if edge_index is not None:
            out = self.model(x, edge_index)
        else:
            out = self.model(x)

        # Compute loss on training nodes
        loss = self.criterion(out[train_mask], y[train_mask])

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Compute accuracy and F1 score
        pred = out[train_mask].argmax(dim=1)
        y_true = y[train_mask].cpu().numpy()
        y_pred = pred.cpu().numpy()

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')

        return loss.item(), acc, f1

    def evaluate(self, data: Data, mask: torch.Tensor) -> Tuple[float, float, float, Dict[str, float]]:
        self.model.eval()

        # Move data to device
        x = data.x.to(self.device)
        y = data.y.to(self.device)
        edge_index = data.edge_index.to(self.device) if hasattr(data, 'edge_index') else None
        mask = mask.to(self.device)

        with torch.no_grad():
            # Forward pass
            if edge_index is not None:
                out = self.model(x, edge_index)
            else:
                out = self.model(x)

            # Compute loss
            loss = self.criterion(out[mask], y[mask])

            # Compute predictions
            pred = out[mask].argmax(dim=1)
            y_true = y[mask].cpu().numpy()
            y_pred = pred.cpu().numpy()

            # Compute metrics
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')

            # Compute per-class metrics
            class_f1 = f1_score(y_true, y_pred, average=None)
            class_precision = precision_score(y_true, y_pred, average=None)
            class_recall = recall_score(y_true, y_pred, average=None)

            # Create metrics dictionary
            metrics = {
                'loss': loss.item(),
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'class_f1': class_f1,
                'class_precision': class_precision,
                'class_recall': class_recall
            }

        return loss.item(), acc, f1, metrics

    def train(self, data: Data, epochs: int = 100) -> Dict[str, List[float]]:
        """
        Train the model
        """
        logger.info(f"Starting training for {epochs} epochs")
        start_time = time.time()

        for epoch in range(epochs):
            # Train for one epoch
            train_loss, train_acc, train_f1 = self.train_epoch(data)

            # Evaluate on validation set
            val_loss, val_acc, val_f1, _ = self.evaluate(data, data.val_mask)

            # Log progress
            logger.info(f"Epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
                        f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            # Update scheduler if provided
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0

                # Save model
                self.save_model()

                logger.info(f"New best model at epoch {epoch + 1} with validation accuracy: {val_acc:.4f}")
            else:
                self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1} after {self.patience} epochs without improvement")
                    break

        end_time = time.time()
        total_time = end_time - start_time

        logger.info(f"Training completed in {total_time:.2f} seconds")
        logger.info(f"Best model at epoch {self.best_epoch + 1} with validation accuracy: {self.best_val_acc:.4f}")

        # Load best model
        self.load_model()

        return self.history

    def test(self, data: Data) -> Dict[str, float]:
        """
        Test the model
        """
        # Load best model if not already loaded
        self.load_model()

        # Evaluate on test set
        _, test_acc, test_f1, test_metrics = self.evaluate(data, data.test_mask)

        logger.info(f"Test results - accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")

        return test_metrics

    def predict(self, data: Data, mask: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Make predictions
        """
        self.model.eval()

        # Move data to device
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device) if hasattr(data, 'edge_index') else None

        with torch.no_grad():
            # Forward pass
            if edge_index is not None:
                out = self.model(x, edge_index)
            else:
                out = self.model(x)

            # Apply mask if provided
            if mask is not None:
                mask = mask.to(self.device)
                out = out[mask]

            # Get predictions
            pred = out.argmax(dim=1).cpu().numpy()

        return pred

    def save_model(self, path: Optional[str] = None):
        """
        Save model checkpoint
        """
        if path is None:
            path = os.path.join(self.model_dir, f"{self.model_name}_best.pt")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'history': self.history
        }, path)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: Optional[str] = None):
        """
        Load model checkpoint
        """
        if path is None:
            path = os.path.join(self.model_dir, f"{self.model_name}_best.pt")

        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_val_loss = checkpoint['best_val_loss']
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch = checkpoint['best_epoch']

            # Load history if available
            if 'history' in checkpoint:
                self.history = checkpoint['history']

            logger.info(f"Model loaded from {path}")
        else:
            logger.warning(f"Model checkpoint not found at {path}")