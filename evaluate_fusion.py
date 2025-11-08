# ─────────────────────────────────────────────────────────────
# evaluate_fusion.py → ONE COMMAND → ALL METRICS FOR TECHNICAL PAPER
# ─────────────────────────────────────────────────────────────
# Run with:  python evaluate_fusion.py --pred_folder results_pred --true_csv ground_truth.csv

import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import accuracy_score, jaccard_score
import argparse

VOIDS = 0
BACKGROUND = 1
FABRIC = 2


def load_mask(path):
    mask = np.load(path) if path.endswith('.npz') else np.asarray(Image.open(path))
    return mask


def composition_accuracy(pred_composition, true_composition):
    """Returns % of correctly predicted material ratios (100% if perfect)"""
    return 100.0 * (1.0 - np.mean(np.abs(pred_composition - true_composition)))


parser = argparse.ArgumentParser()
parser.add_argument('--pred_folder', default='results_pred')
parser.add_argument('--true_csv', default='data/ground_truth_composition.csv')
args = parser.parse_args()

# 1. Load ground truth compositions
true_df = pd.read_csv(args.true_csv)
true_comp = {row['filename']: np.array([row['cotton'], row['polyester'], row['elastaan'],
                                        row['antistatic'], row['modacrylic'], row['pla']]) / 100.0
             for _, row in true_df.iterrows()}

# 2. Find all prediction masks
pred_files = glob(os.path.join(args.pred_folder, "*.png")) + glob(os.path.join(args.pred_folder, "*.npz"))

results = []

for pred_path in pred_files:
    filename = os.path.basename(pred_path).split('_')[0] + '.hsimage'  # sample004_pred.png → sample004

    # Load predicted segmentation mask
    pred_mask = np
    pil.open(pred_path).convert('L')
    pred_mask = np.array(pred_mask)
    pred_mask = np.where(pred_mask > 200, FABRIC, np.where(pred_mask > 50, BACKGROUND, VOIDS))

    # Load TRUE segmentation mask (from your original masks)
    true_mask_path = f"data/masks/{filename.replace('.hsimage', '')}_mask.png"
    true_mask = np
    pil.open(true_mask_path).convert('L')
    true_mask = np.array(true_mask)
    true_mask = np.where(true_mask > 200, FABRIC, np.where(true_mask > 50, BACKGROUND, VOIDS))

    # ───── PIXEL ACCURACY & IoU ─────
    pixel_acc = accuracy_score(true_mask.ravel(), pred_mask.ravel()) * 100
    iou = jaccard_score(true_mask.ravel(), pred_mask.ravel(), average=None)[FABRIC]

    # ───── COMPOSITION PREDICTION ─────
    fabric_pixels = pred_mask == FABRIC
    if fabric_pixels.sum() == 0:
        pred_comp = np.zeros(6)
    else:
        # Here we use your already saved .npy composition map
        comp_map = np.load(pred_path.replace('.png', '_composition.npy'))  # shape (H,W,6)
        pred_comp = comp_map[fabric_pixels].mean(axis=0)
        pred_comp = pred_comp / pred_comp.sum()  # renormalize

    true_comp_vec = true_comp.get(filename, np.zeros(6))
    mae = np.mean(np.abs(pred_comp - true_comp_vec))
    mse = np.mean((pred_comp - true_comp_vec) ** 2)
    cosine = np.dot(pred_comp, true_comp_vec) / (np.linalg.norm(pred_comp) * np.linalg.norm(true_comp_vec) + 1e-8)
    comp_acc = composition_accuracy(pred_comp, true_comp_vec)

    results.append({
        'file': filename.replace('.hsimage', ''),
        'cosine': round(cosine, 4),
        'mae': round(mae, 4),
        'mse': round(mse, 4),
        'pixel_acc_%': round(pixel_acc, 1),
        'fabric_iou': round(iou, 3),
        'comp_acc_%': round(comp_acc, 1)
    })

# ───── FINAL TABLES (copy-paste into paper) ─────
df = pd.DataFrame(results)
print("\nCOPY-PASTE INTO TECHNICAL PAPER:\n")
print(df[['file', 'cosine', 'mae', 'mse', 'pixel_acc_%', 'fabric_iou', 'comp_acc_%']].to_markdown(index=False))

print("\nAVERAGES:")
print(f"Mean Cosine Similarity : {df['cosine'].mean():.4f}")
print(f"Mean MAE              : {df['mae'].mean():.4f}")
print(f"Mean Pixel Accuracy   : {df['pixel_acc_%'].mean():.1f}%")
print(f"Mean Fabric IoU       : {df['fabric_iou'].mean():.3f}")
print(f"Mean Composition Acc  : {df['comp_acc_%'].mean():.1f}%")

# Auto-save LaTeX table
latex = df[['file', 'cosine', 'mae', 'mse', 'pixel_acc_%', 'fabric_iou', 'comp_acc_%']].to_latex(index=False,
                                                                                                 float_format="%.3f")
with open("evaluation_table.tex", "w") as f:
    f.write(latex)
print("\n→ evaluation_table.tex saved for your Technical Paper!")