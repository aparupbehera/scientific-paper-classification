import os
import torch
import logging
import pandas as pd
import config
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.models import TextOnlyBaseline, GraphOnlyBaseline, HybridGNN
from src.trainer import Trainer
from src.visualization import Visualization
from src.utils import setup_logging, set_seed, get_available_device


def train_text_model(text_data, output_dir):
    """
    Train text-only model
    """
    logger = logging.getLogger()
    logger.info("Training text-only model")

    # Get available device
    device = get_available_device()

    # Create model
    model = TextOnlyBaseline(
        input_dim=text_data.x.shape[1],
        hidden_dim=config.HIDDEN_DIM,
        output_dim=text_data.num_classes,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)

    # Create optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        model_name='text_only',
        output_dir=output_dir,
        patience=config.PATIENCE,
        scheduler=scheduler
    )

    # Train model
    history = trainer.train(text_data, epochs=config.EPOCHS)

    # Test model
    test_metrics = trainer.test(text_data)

    # Visualize results if enabled
    if config.VISUALIZE:
        visualizer = Visualization(os.path.join(output_dir, 'figures'))
        visualizer.plot_training_history(
            history=history,
            metrics=['loss', 'accuracy', 'f1'],
            title='Text-Only Model Training History',
            filename='text_only_training_history.png'
        )

        # Get predictions on test set
        y_pred = trainer.predict(text_data, text_data.test_mask)
        y_true = text_data.y[text_data.test_mask].cpu().numpy()

        # Visualize confusion matrix
        class_names = [str(i) for i in range(text_data.num_classes)]
        visualizer.plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            title='Text-Only Model Confusion Matrix',
            filename='text_only_confusion_matrix.png'
        )

    return test_metrics


def train_graph_model(graph_data, output_dir):
    """
    Train graph-only model
    """
    logger = logging.getLogger()
    logger.info("Training graph-only model")

    # Get available device
    device = get_available_device()

    # Create model
    model = GraphOnlyBaseline(
        input_dim=graph_data.x.shape[1],
        hidden_dim=config.HIDDEN_DIM,
        output_dim=graph_data.num_classes,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)

    # Create optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        model_name='graph_only',
        output_dir=output_dir,
        patience=config.PATIENCE,
        scheduler=scheduler
    )

    # Train model
    history = trainer.train(graph_data, epochs=config.EPOCHS)

    # Test model
    test_metrics = trainer.test(graph_data)

    # Visualize results if enabled
    if config.VISUALIZE:
        visualizer = Visualization(os.path.join(output_dir, 'figures'))
        visualizer.plot_training_history(
            history=history,
            metrics=['loss', 'accuracy', 'f1'],
            title='Graph-Only Model Training History',
            filename='graph_only_training_history.png'
        )

        # Get predictions on test set
        y_pred = trainer.predict(graph_data, graph_data.test_mask)
        y_true = graph_data.y[graph_data.test_mask].cpu().numpy()

        # Visualize confusion matrix
        class_names = [str(i) for i in range(graph_data.num_classes)]
        visualizer.plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            title='Graph-Only Model Confusion Matrix',
            filename='graph_only_confusion_matrix.png'
        )

    return test_metrics


def train_hybrid_model(hybrid_data, output_dir):
    """
    Train hybrid model
    """
    logger = logging.getLogger()
    logger.info("Training hybrid model")

    # Get available device
    device = get_available_device()

    # Extract text and graph dimensions if available
    text_dim = hybrid_data.text_dim if hasattr(hybrid_data, 'text_dim') else hybrid_data.x.shape[1] // 2
    graph_dim = hybrid_data.graph_dim if hasattr(hybrid_data, 'graph_dim') else hybrid_data.x.shape[1] // 2

    # Create model
    model = HybridGNN(
        text_dim=text_dim,
        graph_dim=graph_dim,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=hybrid_data.num_classes,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        gnn_type=config.GNN_TYPE,
        combination=config.COMBINATION
    ).to(device)

    # Create optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        model_name='hybrid',
        output_dir=output_dir,
        patience=config.PATIENCE,
        scheduler=scheduler
    )

    # Train model
    history = trainer.train(hybrid_data, epochs=config.EPOCHS)

    # Test model
    test_metrics = trainer.test(hybrid_data)

    # Visualize results if enabled
    if config.VISUALIZE:
        visualizer = Visualization(os.path.join(output_dir, 'figures'))
        visualizer.plot_training_history(
            history=history,
            metrics=['loss', 'accuracy', 'f1'],
            title='Hybrid Model Training History',
            filename='hybrid_training_history.png'
        )

        # Get predictions on test set
        y_pred = trainer.predict(hybrid_data, hybrid_data.test_mask)
        y_true = hybrid_data.y[hybrid_data.test_mask].cpu().numpy()

        # Visualize confusion matrix
        class_names = [str(i) for i in range(hybrid_data.num_classes)]
        visualizer.plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            title='Hybrid Model Confusion Matrix',
            filename='hybrid_confusion_matrix.png'
        )

    return test_metrics


def main():
    # Set up logging
    logger = setup_logging(config.LOGS_DIR,
                           log_level=logging.DEBUG if config.VERBOSE else logging.INFO)

    # Set random seed
    set_seed(config.SEED)

    logger.info("Starting model training")

    # Dictionary to store metrics for each model
    all_metrics = {}

    # Train models according to config
    for model_name in config.MODELS_TO_TRAIN:
        # Train text-only model
        if model_name == 'text':
            text_data_path = os.path.join(config.PROCESSED_DIR, 'text_data.pt')
            if os.path.exists(text_data_path):
                text_data = torch.load(text_data_path)
                text_metrics = train_text_model(text_data, config.OUTPUT_DIR)
                all_metrics['text'] = text_metrics
            else:
                logger.error(f"Text data not found at {text_data_path}")

        # Train graph-only model
        if model_name == 'graph':
            graph_data_path = os.path.join(config.PROCESSED_DIR, 'graph_data.pt')
            if os.path.exists(graph_data_path):
                graph_data = torch.load(graph_data_path)
                graph_metrics = train_graph_model(graph_data, config.OUTPUT_DIR)
                all_metrics['graph'] = graph_metrics
            else:
                logger.error(f"Graph data not found at {graph_data_path}")

        # Train hybrid model
        if model_name == 'hybrid':
            hybrid_data_path = os.path.join(config.PROCESSED_DIR, 'hybrid_data.pt')
            if os.path.exists(hybrid_data_path):
                hybrid_data = torch.load(hybrid_data_path)
                hybrid_metrics = train_hybrid_model(hybrid_data, config.OUTPUT_DIR)
                all_metrics['hybrid'] = hybrid_metrics
            else:
                logger.error(f"Hybrid data not found at {hybrid_data_path}")

    # Compare model performance if multiple models were trained
    if len(all_metrics) > 1:
        logger.info("Comparing model performance")

        # Create metrics table
        metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'F1', 'Precision', 'Recall'])

        for model_name, metrics in all_metrics.items():
            metrics_df = metrics_df._append({
                'Model': model_name.capitalize(),
                'Accuracy': metrics['accuracy'],
                'F1': metrics['f1'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall']
            }, ignore_index=True)

        # Save metrics table
        metrics_df.to_csv(os.path.join(config.OUTPUT_DIR, 'model_comparison.csv'), index=False)

        # Visualize comparison if enabled
        if config.VISUALIZE:
            visualizer = Visualization(os.path.join(config.OUTPUT_DIR, 'figures'))
            visualizer.plot_model_comparison(
                results={name: {'accuracy': metrics['accuracy'], 'f1': metrics['f1']}
                        for name, metrics in all_metrics.items()},
                metric_names=['accuracy', 'f1'],
                title='Model Comparison',
                filename='model_comparison.png'
            )

        # Log comparison
        logger.info("\nModel Comparison:")
        logger.info(metrics_df.to_string(index=False))

    logger.info("Training completed successfully")


if __name__ == '__main__':
    main()