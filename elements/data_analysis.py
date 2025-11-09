"""
Data Analysis and Diagnostics for Hyperspectral Dataset
Add this to your project to diagnose training issues
"""

import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_dataset_distribution(dataset, dataset_name="Dataset"):
    """
    Analyze the class distribution in your dataset.
    This helps identify class imbalance issues.

    Args:
        dataset: Your HSI dataset object
        dataset_name: Name for logging purposes
    """
    print(f"\n{'=' * 80}")
    print(f"Analyzing {dataset_name}")
    print(f"{'=' * 80}")

    class_names = dataset.get_class_names()
    class_counts = defaultdict(int)
    total_pixels = 0
    sample_compositions = []

    # Analyze each sample
    for idx in range(len(dataset)):
        try:
            # Get data based on dataset type
            if hasattr(dataset, 'training_mode'):
                dataset.training_mode = True

            data, label = dataset[idx]

            # Convert to numpy if tensor
            if isinstance(label, torch.Tensor):
                label_np = label.numpy()
            else:
                label_np = label

            # Flatten if multi-dimensional
            label_flat = label_np.flatten()

            # Count classes in this sample
            unique, counts = np.unique(label_flat, return_counts=True)
            sample_comp = {}

            for cls, count in zip(unique, counts):
                cls = int(cls)  # Convert to int immediately
                class_counts[cls] += int(count)  # Also convert count to int
                total_pixels += int(count)
                sample_comp[cls] = int(count)

            sample_compositions.append(sample_comp)

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    # Calculate and display overall distribution
    print(f"\nTotal samples: {len(dataset)}")
    print(f"Total pixels analyzed: {total_pixels:,}")
    print(f"\n{'Class Distribution:':-^80}")
    print(f"{'Class ID':<12} {'Class Name':<20} {'Count':<15} {'Percentage':<15}")
    print("-" * 80)

    class_weights = {}
    for cls_id in sorted(class_counts.keys()):
        cls_id = int(cls_id)  # Ensure integer
        count = class_counts[cls_id]
        percentage = (count / total_pixels) * 100
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"Unknown_{cls_id}"
        print(f"{cls_id:<12} {cls_name:<20} {count:<15,} {percentage:<15.2f}%")

        # Calculate weight (inverse frequency, avoid division by zero)
        if count > 0:
            class_weights[cls_id] = total_pixels / (len(class_counts) * count)
        else:
            class_weights[cls_id] = 1.0

    # Display recommended class weights
    print(f"\n{'Recommended Class Weights:':-^80}")
    print("Use these weights in your loss function to handle class imbalance:")
    print(f"class_weights = torch.tensor([", end="")
    for i in range(len(class_names)):
        weight = class_weights.get(i, 1.0)
        print(f"{weight:.4f}", end="")
        if i < len(class_names) - 1:
            print(", ", end="")
    print("])")

    # Analyze sample-level statistics
    print(f"\n{'Sample-Level Analysis:':-^80}")
    samples_per_class = defaultdict(int)
    for sample_comp in sample_compositions:
        for cls in sample_comp.keys():
            samples_per_class[int(cls)] += 1

    print(f"{'Class ID':<12} {'Class Name':<20} {'Samples with this class':<25}")
    print("-" * 80)
    for cls_id in sorted(samples_per_class.keys()):
        cls_id = int(cls_id)  # Ensure integer
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"Unknown_{cls_id}"
        print(f"{cls_id:<12} {cls_name:<20} {samples_per_class[cls_id]:<25}")

    print(f"{'=' * 80}\n")

    return class_weights, class_counts


def visualize_class_distribution(class_counts, class_names, save_path=None):
    """
    Create a bar chart visualization of class distribution.

    Args:
        class_counts: Dictionary of class counts
        class_names: List of class names
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    labels = [class_names[c] if c < len(class_names) else f"Class_{c}" for c in classes]

    bars = ax.bar(range(len(classes)), counts, color='steelblue', alpha=0.7)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Pixel Count', fontsize=12)
    ax.set_title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add percentage labels on bars
    total = sum(counts)
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        percentage = (count / total) * 100
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{percentage:.1f}%',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    plt.show()


def check_label_consistency(dataset):
    """
    Check if labels are consistent across samples.
    Helps identify annotation issues.
    """
    print(f"\n{'Label Consistency Check:':-^80}")

    label_shapes = set()
    label_dtypes = set()
    label_ranges = []

    for idx in range(min(10, len(dataset))):  # Check first 10 samples
        try:
            _, label = dataset[idx]
            if isinstance(label, torch.Tensor):
                label_np = label.numpy()
            else:
                label_np = label

            label_shapes.add(label_np.shape)
            label_dtypes.add(str(label_np.dtype))
            label_ranges.append((label_np.min(), label_np.max()))
        except Exception as e:
            print(f"Error checking sample {idx}: {e}")

    print(f"Label shapes found: {label_shapes}")
    print(f"Label dtypes found: {label_dtypes}")
    print(f"Label value ranges (first 10 samples): {label_ranges}")

    if len(label_shapes) > 1:
        print("⚠️  WARNING: Inconsistent label shapes detected!")
    if len(label_dtypes) > 1:
        print("⚠️  WARNING: Inconsistent label data types detected!")

    print(f"{'=' * 80}\n")


def convert_onehot_to_indices(label):
    """
    Convert one-hot encoded labels to class indices.

    Args:
        label: One-hot tensor of shape (C, H, W) or (1, C, H, W)

    Returns:
        Class indices tensor of shape (H, W)
    """
    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label)

    # Remove batch dimension if present
    if label.dim() == 4 and label.shape[0] == 1:
        label = label.squeeze(0)

    # Convert one-hot (C, H, W) to indices (H, W)
    if label.dim() == 3:
        class_indices = torch.argmax(label, dim=0)
    else:
        class_indices = label

    return class_indices.long()


# Usage example
if __name__ == "__main__":
    """
    Add this to your hit_pipeline.py after loading the dataset:

    from data_analysis import analyze_dataset_distribution, visualize_class_distribution, check_label_consistency

    # After: dataset = load_hsi_dataset(...)
    check_label_consistency(dataset)
    class_weights, class_counts = analyze_dataset_distribution(dataset, "Training Dataset")
    visualize_class_distribution(class_counts, dataset.get_class_names(), 
                                 save_path="log/experiment/class_distribution.png")
    """
    print("Import this module into your hit_pipeline.py to use the analysis functions.")