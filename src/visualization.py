import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Dict, Optional
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


class Visualization:
    """
    Class for visualizing dataset statistics and model results
    """

    def __init__(self, output_dir: str = './figures'):
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def plot_category_distribution(self,
                                   category_counts,
                                   title: str = 'Category Distribution',
                                   top_n: int = 20,
                                   filename: str = 'category_distribution.png'):
        """
        Plot the distribution of paper categories
        """
        plt.figure(figsize=(14, 7))

        # Get top N categories
        top_categories = category_counts.head(top_n)

        # Create barplot
        sns.barplot(x=top_categories.index, y=top_categories.values)
        plt.title(title)
        plt.xticks(rotation=90)
        plt.ylabel('Number of Papers')
        plt.tight_layout()

        # Save figure
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

        logger.info(f"Saved category distribution plot to {os.path.join(self.output_dir, filename)}")

    def plot_citation_distribution(self,
                                   graph,
                                   title: str = 'Citation Distribution',
                                   plot_type: str = 'both',
                                   filename: str = 'citation_distribution.png'):
        """
        Plot the distribution of citations
        """
        if plot_type == 'both':
            plt.figure(figsize=(14, 6))

            # In-degree distribution
            plt.subplot(1, 2, 1)
            in_degrees = [d for n, d in graph.in_degree()]
            sns.histplot(in_degrees, bins=50, kde=True)
            plt.title('In-Degree Distribution')
            plt.xlabel('Number of Citations Received')
            plt.ylabel('Count')

            # Out-degree distribution
            plt.subplot(1, 2, 2)
            out_degrees = [d for n, d in graph.out_degree()]
            sns.histplot(out_degrees, bins=50, kde=True)
            plt.title('Out-Degree Distribution')
            plt.xlabel('Number of Citations Made')
            plt.ylabel('Count')

            plt.suptitle(title)
            plt.tight_layout()

        elif plot_type == 'in':
            plt.figure(figsize=(10, 6))
            in_degrees = [d for n, d in graph.in_degree()]
            sns.histplot(in_degrees, bins=50, kde=True)
            plt.title(f'{title} - In-Degree')
            plt.xlabel('Number of Citations Received')
            plt.ylabel('Count')
            plt.tight_layout()

        elif plot_type == 'out':
            plt.figure(figsize=(10, 6))
            out_degrees = [d for n, d in graph.out_degree()]
            sns.histplot(out_degrees, bins=50, kde=True)
            plt.title(f'{title} - Out-Degree')
            plt.xlabel('Number of Citations Made')
            plt.ylabel('Count')
            plt.tight_layout()

        # Save figure
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

        logger.info(f"Saved citation distribution plot to {os.path.join(self.output_dir, filename)}")

    def plot_embeddings_tsne(self,
                             embeddings,
                             labels=None,
                             label_names: Optional[List[str]] = None,
                             perplexity: int = 30,
                             n_components: int = 2,
                             title: str = 'Embeddings t-SNE Visualization',
                             filename: str = 'embeddings_tsne.png',
                             max_points: int = 5000,
                             random_state: int = 42):
        """
        Plot t-SNE visualization of embeddings
        """
        # Convert to numpy
        embeddings = np.array(embeddings)
        if labels is not None:
            labels = np.array(labels)

        # Subsample if too many points
        if embeddings.shape[0] > max_points:
            logger.info(f"Subsampling {max_points} points from {embeddings.shape[0]} for t-SNE visualization")
            indices = np.random.choice(embeddings.shape[0], max_points, replace=False)
            embeddings = embeddings[indices]
            if labels is not None:
                labels = labels[indices]

        # Apply t-SNE
        logger.info(f"Applying t-SNE with perplexity={perplexity}...")
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Create plot
        plt.figure(figsize=(12, 10))

        if n_components == 2:
            if labels is not None:
                # Create scatter plot with labels
                scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)

                # Add legend
                if label_names is not None:
                    # Create custom legend
                    handles = []
                    for i, name in enumerate(label_names):
                        handle = plt.Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor=scatter.cmap(scatter.norm(i)),
                                            markersize=10, label=name)
                        handles.append(handle)

                    # Show legend with specified names
                    plt.legend(handles=handles, loc='best', title='Categories')
                else:
                    # Add colorbar instead of legend for many classes
                    plt.colorbar(scatter, label='Category')
            else:
                # Create scatter plot without labels
                plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)

            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')

        elif n_components == 3:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            if labels is not None:
                scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], embeddings_2d[:, 2],
                                     c=labels, cmap='viridis', alpha=0.7)

                # Add colorbar
                plt.colorbar(scatter, ax=ax, label='Category')
            else:
                ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], embeddings_2d[:, 2], alpha=0.7)

            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
            ax.set_zlabel('t-SNE Component 3')

        plt.title(title)
        plt.tight_layout()

        # Save figure
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

        logger.info(f"Saved t-SNE visualization to {os.path.join(self.output_dir, filename)}")

    def plot_confusion_matrix(self,
                              y_true,
                              y_pred,
                              class_names: Optional[List[str]] = None,
                              normalize: bool = True,
                              title: str = 'Confusion Matrix',
                              filename: str = 'confusion_matrix.png'):
        """
        Plot confusion matrix
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'

        # Create plot
        plt.figure(figsize=(12, 10))

        # Plot with Seaborn
        if class_names is not None and len(class_names) <= 20:
            sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
        else:
            sns.heatmap(cm, annot=False, cmap='Blues')

        plt.title(title)
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()

        # Save figure
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

        logger.info(f"Saved confusion matrix to {os.path.join(self.output_dir, filename)}")

    def plot_training_history(self,
                              history: Dict[str, List[float]],
                              metrics: List[str] = ['loss', 'accuracy'],
                              title: str = 'Training History',
                              filename: str = 'training_history.png'):
        """
        Plot training history
        """
        plt.figure(figsize=(12, 6))

        # Create subplots for each metric
        for i, metric in enumerate(metrics):
            if metric in history and f'val_{metric}' in history:
                plt.subplot(1, len(metrics), i + 1)
                plt.plot(history[metric], label=f'Training {metric}')
                plt.plot(history[f'val_{metric}'], label=f'Validation {metric}')
                plt.title(f'{metric.capitalize()}')
                plt.xlabel('Epoch')
                plt.ylabel(metric.capitalize())
                plt.legend()

        plt.suptitle(title)
        plt.tight_layout()

        # Save figure
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

        logger.info(f"Saved training history plot to {os.path.join(self.output_dir, filename)}")

    def plot_model_comparison(self,
                              results: Dict[str, Dict[str, float]],
                              metric_names: List[str] = ['accuracy', 'f1'],
                              title: str = 'Model Comparison',
                              filename: str = 'model_comparison.png'):
        """
        Plot comparison of different models
        """
        # Create DataFrame for plotting
        data = []
        for model_name, metrics in results.items():
            for metric_name in metric_names:
                if metric_name in metrics:
                    data.append({
                        'Model': model_name,
                        'Metric': metric_name.capitalize(),
                        'Value': metrics[metric_name]
                    })

        import pandas as pd
        df = pd.DataFrame(data)

        # Create plot
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Model', y='Value', hue='Metric', data=df)
        plt.title(title)
        plt.ylim(0, 1)
        plt.ylabel('Score')
        plt.tight_layout()

        # Save figure
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

        logger.info(f"Saved model comparison plot to {os.path.join(self.output_dir, filename)}")