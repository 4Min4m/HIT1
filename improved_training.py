"""
Enhanced training functions with detailed metrics and weighted loss
Replace the training functions in hit_pipeline.py
"""

import torch
import time
import numpy as np
from torch.amp import GradScaler, autocast
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings("ignore")


def calculate_class_weights(dataset, ignore_index=0, device='cuda'):
    """
    Calculate class weights based on inverse frequency.

    Args:
        dataset: Training dataset
        ignore_index: Class index to ignore (e.g., void class)
        device: Device to put weights on

    Returns:
        torch.Tensor: Class weights
    """
    print(f"\n{'Calculating Class Weights:':-^80}")

    class_counts = {}
    total_pixels = 0

    # Sample up to 100 samples to estimate distribution
    sample_size = min(100, len(dataset))

    for idx in range(sample_size):
        try:
            _, label = dataset[idx]
            if isinstance(label, torch.Tensor):
                label_np = label.cpu().numpy()
            else:
                label_np = label

            label_flat = label_np.flatten()
            unique, counts = np.unique(label_flat, return_counts=True)

            for cls, count in zip(unique, counts):
                cls = int(cls)
                if cls != ignore_index:
                    class_counts[cls] = class_counts.get(cls, 0) + count
                    total_pixels += count
        except Exception as e:
            print(f"Warning: Skipping sample {idx} due to error: {e}")
            continue

    if not class_counts:
        print("Warning: No valid classes found (excluding ignore_index). Returning default weights.")
        # Attempt to get num_classes from dataset if possible, else default to a small number
        num_classes = getattr(dataset, 'num_classes', 10)
        return torch.ones(num_classes).to(device)

    # Calculate weights (inverse frequency)
    num_classes = max(class_counts.keys()) + 1
    weights = torch.ones(num_classes)

    for cls, count in class_counts.items():
        if count > 0:
            # Inverse frequency with sqrt smoothing
            weights[cls] = np.sqrt(total_pixels / (len(class_counts) * count))

    # Set ignored class weight to 0
    weights[ignore_index] = 0

    # Normalize weights
    weights = weights / weights.sum() * num_classes

    print(f"Class weights calculated from {sample_size} samples:")
    for i, w in enumerate(weights):
        count = class_counts.get(i, 0)
        pct = (count / total_pixels * 100) if total_pixels > 0 else 0
        print(f"  Class {i}: weight={w:.4f}, pixels={count:,} ({pct:.2f}%)")

    print(f"{'=' * 80}\n")

    return weights.to(device)


@torch.no_grad()
def detailed_validation(model, val_loader, loss_func, device, class_names, epoch):
    """
    Perform detailed validation with per-class metrics.

    Returns:
        val_loss: Average validation loss
        metrics: Dictionary of detailed metrics
    """
    model.eval()

    running_val_loss = 0.0
    all_preds = []
    all_labels = []
    running_loss_count = 0

    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        with autocast(device_type=device):
            outputs = model(inputs)
            loss = loss_func(outputs, labels)

        running_val_loss += loss.item()
        running_loss_count += 1

        # Get predictions
        preds = outputs.argmax(dim=1)

        # Flatten for metrics
        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = running_val_loss / running_loss_count

    # Calculate detailed metrics every 10 epochs
    metrics = {}
    if epoch % 10 == 0:
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Per-class accuracy
        unique_classes = np.unique(all_labels)
        per_class_acc = {}

        for cls in unique_classes:
            mask = all_labels == cls
            if mask.sum() > 0:
                acc = (all_preds[mask] == cls).sum() / mask.sum()
                per_class_acc[cls] = acc

        metrics['per_class_accuracy'] = per_class_acc

        # Overall accuracy (excluding ignore_index=0)
        mask = all_labels > 0
        if mask.sum() > 0:
            overall_acc = (all_preds[mask] == all_labels[mask]).sum() / mask.sum()
            metrics['overall_accuracy'] = overall_acc

        # Print detailed report
        print(f"\n{'Validation Metrics (Epoch ' + str(epoch) + '):':-^80}")
        print(f"Overall Accuracy: {metrics.get('overall_accuracy', 0):.4f}")
        print(f"\nPer-Class Accuracy:")
        for cls, acc in sorted(per_class_acc.items()):
            cls_name = class_names[cls] if cls < len(class_names) else f"Class_{cls}"
            print(f"  {cls_name:<20}: {acc:.4f}")
        print(f"{'=' * 80}\n")

    return avg_loss, metrics


def run_training_improved(train_loader, val_loader, model, optimizer, loss_func,
                          scheduler, config, class_names):
    """
    Enhanced training function with detailed metrics and early stopping.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        model: Neural network model
        optimizer: Optimizer
        loss_func: Loss function
        scheduler: Learning rate scheduler
        config: Configuration object
        class_names: List of class names
    """
    from elements.save_model import save_model
    from elements.visualize import show_loss_tb
    from elements.utils import LoggerSingleton
    import os

    logger = LoggerSingleton.get_logger()

    # Initialize tracking variables
    if not hasattr(run_training_improved, 'best_val_loss'):
        run_training_improved.best_val_loss = 9999
        run_training_improved.best_epoch = 0
        run_training_improved.patience_counter = 0

    train_losses = []
    val_losses = []
    scaler = GradScaler()
    start_time = time.time()

    # Early stopping parameters
    early_stop_patience = config.patience * 3  # Stop if no improvement for 3x scheduler patience

    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        running_loss_count = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(config.device)
            # Explicitly cast labels to torch.long
            labels = labels.to(config.device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=config.device):
                outputs = model(inputs)
                loss = loss_func(outputs, labels)


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()
            running_loss_count += 1

        train_loss = running_train_loss / running_loss_count
        train_losses.append(train_loss)

        # Validation phase
        if (epoch + 1) % config.val_frequency == 0 or epoch == config.num_epochs - 1 or epoch == 0:
            val_loss, metrics = detailed_validation(
                model, val_loader, loss_func, config.device, class_names, epoch
            )
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            # Track best model
            model_saved = False
            if val_loss < run_training_improved.best_val_loss:
                run_training_improved.best_val_loss = val_loss
                run_training_improved.best_epoch = epoch
                run_training_improved.patience_counter = 0
                save_model(model, os.path.join(config.writer.log_dir))
                model_saved = True
            else:
                run_training_improved.patience_counter += 1

            # Visualization & logging
            show_loss_tb(val_loss, epoch, writer=config.writer, name='Valid Loss')
            show_loss_tb(train_loss, epoch, writer=config.writer, name='Train Loss')

            # Log per-class accuracy to TensorBoard
            if 'per_class_accuracy' in metrics:
                for cls, acc in metrics['per_class_accuracy'].items():
                    config.writer.add_scalar(f'Val_Accuracy/Class_{cls}', acc, epoch)
                config.writer.add_scalar('Val_Accuracy/Overall',
                                         metrics.get('overall_accuracy', 0), epoch)

            time_passed = time.time() - start_time

            logger.info(f"Epoch {epoch + 1:<3}/{config.num_epochs:<3} | "
                        f"TrainLoss: {train_loss:<8.6f} | "
                        f"ValidLoss: {val_loss:<8.6f} | "
                        f"LR: {optimizer.param_groups[0]['lr']:<8.6f} | "
                        f"Saved: {str(model_saved):<5} | "
                        f"Best@Epoch: {run_training_improved.best_epoch + 1:<3} | "
                        f"Time: {time_passed:<6.2f}s")

            # Early stopping check
            if run_training_improved.patience_counter >= early_stop_patience:
                logger.info(f"\n{'=' * 80}")
                logger.info(f"Early stopping triggered! No improvement for {early_stop_patience} epochs.")
                logger.info(
                    f"Best validation loss: {run_training_improved.best_val_loss:.6f} at epoch {run_training_improved.best_epoch + 1}")
                logger.info(f"{'=' * 80}\n")
                break
        else:
            val_losses.append(None)

    # Final summary
    logger.info(f"\n{'Training Complete':-^80}")
    logger.info(f"Best validation loss: {run_training_improved.best_val_loss:.6f}")
    logger.info(f"Best epoch: {run_training_improved.best_epoch + 1}")
    logger.info(f"Total training time: {time.time() - start_time:.2f}s")
    logger.info(f"{'=' * 80}\n")