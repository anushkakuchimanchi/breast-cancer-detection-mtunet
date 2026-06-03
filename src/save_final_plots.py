import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                              precision_recall_curve, classification_report)
from sklearn.preprocessing import label_binarize
import seaborn as sns

# ── CONFIG ───────────────────────────────────────────────────────────────
RUN_PATH = "runs/20260501_124023_MTUNetPlusPlus_24_alpha_0.85_batch_2_benign_malignant_normal"
N_FOLDS = 4
CLASS_NAMES = ['normal', 'benign', 'malignant']
PLOTS_PATH = Path(RUN_PATH) / "final_plots"
PLOTS_PATH.mkdir(parents=True, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────

# ── LOAD ALL FOLD DATA ───────────────────────────────────────────────────
all_metrics = []
all_ens_preds, all_ens_gt = [], []
all_qsvm_preds, all_qsvm_gt = [], []

for i in range(N_FOLDS):
    fold_path = Path(RUN_PATH) / f"fold_{i}"
    try:
        fm = pd.read_csv(fold_path / "metrics.csv")
        all_metrics.append(fm)
    except:
        pass
    try:
        all_ens_preds.append(np.load(fold_path / "ensemble_preds.npy"))
        all_ens_gt.append(np.load(fold_path / "ensemble_gt.npy"))
    except:
        pass
    try:
        all_qsvm_preds.append(np.load(fold_path / "qsvm_preds.npy"))
        all_qsvm_gt.append(np.load(fold_path / "qsvm_gt.npy"))
    except:
        pass

ens_preds = np.concatenate(all_ens_preds)
ens_gt    = np.concatenate(all_ens_gt)
print(f"Loaded {len(ens_gt)} ensemble samples across {N_FOLDS} folds")

# ── 1. LOSS CURVES (avg across folds) ────────────────────────────────────
print("Saving loss curves...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for fm in all_metrics:
    axes[0].plot(fm['Train_loss'], alpha=0.4, color='blue')
    axes[0].plot(fm['Validation_loss'], alpha=0.4, color='orange')
axes[0].set_title('Loss Curves (all folds)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend(['Train', 'Val'])

for fm in all_metrics:
    axes[1].plot(fm['Train_dice'], alpha=0.4, color='green')
    axes[1].plot(fm['Validation_dice'], alpha=0.4, color='red')
axes[1].set_title('Dice Score Curves (all folds)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Dice')
axes[1].legend(['Train', 'Val'])

for fm in all_metrics:
    axes[2].plot(fm['Train_acc'], alpha=0.4, color='purple')
    axes[2].plot(fm['Validation_acc'], alpha=0.4, color='brown')
axes[2].set_title('Accuracy Curves (all folds)')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Accuracy')
axes[2].legend(['Train', 'Val'])

plt.tight_layout()
plt.savefig(PLOTS_PATH / '1_loss_dice_acc_curves.png', dpi=150)
plt.close()
print("  Saved: 1_loss_dice_acc_curves.png")

# ── 2. CONFUSION MATRIX — MTUNet++ ───────────────────────────────────────
print("Saving confusion matrices...")

# load MTUNet++ predictions from results csv
mtunet_gt, mtunet_pred = [], []
for i in range(N_FOLDS):
    fold_path = Path(RUN_PATH) / f"fold_{i}"
    try:
        df = pd.read_csv(fold_path / "results_classification.csv")
        mtunet_gt.extend(df['ground_truth'].tolist())
        mtunet_pred.extend(df['predicted_label'].tolist())
    except Exception as e:
        print(f"  Warning fold {i}: {e}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# MTUNet++ confusion matrix
if mtunet_gt:
    cm = confusion_matrix(mtunet_gt, mtunet_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title('MTUNet++ Softmax\nConfusion Matrix (%)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

# Ensemble confusion matrix
cm_ens = confusion_matrix(ens_gt, ens_preds)
cm_ens_pct = cm_ens.astype(float) / cm_ens.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(cm_ens_pct, annot=True, fmt='.1f', cmap='Greens',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
axes[1].set_title('MTUNet++ + Ensemble SVM\nConfusion Matrix (%)')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig(PLOTS_PATH / '2_confusion_matrices.png', dpi=150)
plt.close()
print("  Saved: 2_confusion_matrices.png")

# ── 3. ROC CURVES ────────────────────────────────────────────────────────
print("Saving ROC curves...")
ens_gt_bin = label_binarize(ens_gt, classes=[0, 1, 2])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = ['blue', 'green', 'red']

# MTUNet++ ROC — need probabilities, use one-hot from pred as proxy
if mtunet_gt:
    mtunet_gt_bin = label_binarize(mtunet_gt, classes=[0, 1, 2])
    mtunet_pred_bin = label_binarize(mtunet_pred, classes=[0, 1, 2])
    for cls_idx, (color, name) in enumerate(zip(colors, CLASS_NAMES)):
        fpr, tpr, _ = roc_curve(mtunet_gt_bin[:, cls_idx],
                                  mtunet_pred_bin[:, cls_idx])
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=color,
                    label=f'{name} (AUC={roc_auc:.2f})')
    axes[0].plot([0,1],[0,1],'k--')
    axes[0].set_title('MTUNet++ Softmax ROC')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend()

# Ensemble ROC
ens_pred_bin = label_binarize(ens_preds, classes=[0, 1, 2])
for cls_idx, (color, name) in enumerate(zip(colors, CLASS_NAMES)):
    fpr, tpr, _ = roc_curve(ens_gt_bin[:, cls_idx],
                              ens_pred_bin[:, cls_idx])
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color=color,
                label=f'{name} (AUC={roc_auc:.2f})')
axes[1].plot([0,1],[0,1],'k--')
axes[1].set_title('MTUNet++ + Ensemble SVM ROC')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend()

plt.tight_layout()
plt.savefig(PLOTS_PATH / '3_roc_curves.png', dpi=150)
plt.close()
print("  Saved: 3_roc_curves.png")

# ── 4. PRECISION RECALL CURVES ───────────────────────────────────────────
print("Saving precision-recall curves...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

if mtunet_gt:
    for cls_idx, (color, name) in enumerate(zip(colors, CLASS_NAMES)):
        prec, rec, _ = precision_recall_curve(mtunet_gt_bin[:, cls_idx],
                                               mtunet_pred_bin[:, cls_idx])
        axes[0].plot(rec, prec, color=color, label=name)
    axes[0].set_title('MTUNet++ Softmax\nPrecision-Recall')
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].legend()

for cls_idx, (color, name) in enumerate(zip(colors, CLASS_NAMES)):
    prec, rec, _ = precision_recall_curve(ens_gt_bin[:, cls_idx],
                                           ens_pred_bin[:, cls_idx])
    axes[1].plot(rec, prec, color=color, label=name)
axes[1].set_title('MTUNet++ + Ensemble SVM\nPrecision-Recall')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].legend()

plt.tight_layout()
plt.savefig(PLOTS_PATH / '4_precision_recall_curves.png', dpi=150)
plt.close()
print("  Saved: 4_precision_recall_curves.png")

# ── 5. T-SNE ─────────────────────────────────────────────────────────────
print("Saving t-SNE visualization (this may take a few mins)...")
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

all_feats, all_labs = [], []
for i in range(N_FOLDS):
    fold_path = Path(RUN_PATH) / f"fold_{i}"
    try:
        # load ensemble gt as labels
        gt = np.load(fold_path / "ensemble_gt.npy")
        # recreate features from saved data — use ensemble_preds shape as proxy
        # we need actual features — check if saved
        feat_file = fold_path / "test_features.npy"
        if feat_file.exists():
            feats = np.load(feat_file)
            all_feats.append(feats)
            all_labs.append(gt)
    except:
        pass

if all_feats:
    X_all = np.vstack(all_feats)
    y_all = np.concatenate(all_labs).astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    for cls_idx, (color, name) in enumerate(zip(colors, CLASS_NAMES)):
        mask = y_all == cls_idx
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=color, label=name, alpha=0.7, s=30)
    plt.title('t-SNE Visualization of MTUNet++ Features')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / '5_tsne.png', dpi=150)
    plt.close()
    print("  Saved: 5_tsne.png")
else:
    print("  Skipped t-SNE: no feature files found (will save on next training run)")




# ── 7. F1 SCORE BAR CHART ────────────────────────────────────────────────
print("Saving F1 score comparison...")
from sklearn.metrics import f1_score

# MTUNet++ F1 scores
mtunet_f1_normal    = f1_score(mtunet_gt, mtunet_pred, labels=[0], average='macro')
mtunet_f1_benign    = f1_score(mtunet_gt, mtunet_pred, labels=[1], average='macro')
mtunet_f1_malignant = f1_score(mtunet_gt, mtunet_pred, labels=[2], average='macro')
mtunet_f1_weighted  = f1_score(mtunet_gt, mtunet_pred, average='weighted')

# Ensemble F1 scores
ens_f1_normal    = f1_score(ens_gt, ens_preds, labels=[0], average='macro')
ens_f1_benign    = f1_score(ens_gt, ens_preds, labels=[1], average='macro')
ens_f1_malignant = f1_score(ens_gt, ens_preds, labels=[2], average='macro')
ens_f1_weighted  = f1_score(ens_gt, ens_preds, average='weighted')

categories = ['Normal', 'Benign', 'Malignant', 'Weighted']
mtunet_scores = [mtunet_f1_normal, mtunet_f1_benign,
                 mtunet_f1_malignant, mtunet_f1_weighted]
ens_scores    = [ens_f1_normal, ens_f1_benign,
                 ens_f1_malignant, ens_f1_weighted]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, mtunet_scores, width,
               label='MTUNet++ Softmax', color='steelblue')
bars2 = ax.bar(x + width/2, ens_scores, width,
               label='MTUNet++ + SVM Ensemble', color='coral')

ax.set_ylabel('F1 Score')
ax.set_title('F1 Score Comparison: MTUNet++ vs SVM Ensemble')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(0, 1.0)
ax.legend()

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_PATH / '7_f1_comparison.png', dpi=150)
plt.close()
print("  Saved: 7_f1_comparison.png")

# ── 8. FINAL OVERALL EXCEL ───────────────────────────────────────────────
print("Saving final overall Excel...")
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ── MTUNet++ Overall Metrics ──
mtunet_overall = {
    'Method': 'MTUNetPlusPlus',
    'Accuracy': round(accuracy_score(mtunet_gt, mtunet_pred), 4),
    'F1_Weighted': round(f1_score(mtunet_gt, mtunet_pred, average='weighted'), 4),
    'F1_Macro': round(f1_score(mtunet_gt, mtunet_pred, average='macro'), 4),
    'F1_Normal': round(f1_score(mtunet_gt, mtunet_pred, labels=[0], average='macro'), 4),
    'F1_Benign': round(f1_score(mtunet_gt, mtunet_pred, labels=[1], average='macro'), 4),
    'F1_Malignant': round(f1_score(mtunet_gt, mtunet_pred, labels=[2], average='macro'), 4),
    'Precision_Weighted': round(precision_score(mtunet_gt, mtunet_pred, average='weighted', zero_division=0), 4),
    'Precision_Normal': round(precision_score(mtunet_gt, mtunet_pred, labels=[0], average='macro', zero_division=0), 4),
    'Precision_Benign': round(precision_score(mtunet_gt, mtunet_pred, labels=[1], average='macro', zero_division=0), 4),
    'Precision_Malignant': round(precision_score(mtunet_gt, mtunet_pred, labels=[2], average='macro', zero_division=0), 4),
    'Recall_Weighted': round(recall_score(mtunet_gt, mtunet_pred, average='weighted', zero_division=0), 4),
    'Recall_Normal': round(recall_score(mtunet_gt, mtunet_pred, labels=[0], average='macro', zero_division=0), 4),
    'Recall_Benign': round(recall_score(mtunet_gt, mtunet_pred, labels=[1], average='macro', zero_division=0), 4),
    'Recall_Malignant': round(recall_score(mtunet_gt, mtunet_pred, labels=[2], average='macro', zero_division=0), 4),
}

# ── Ensemble Overall Metrics ──
ens_overall = {
    'Method': 'MTUNetPlusPlus + SVM Ensemble',
    'Accuracy': round(accuracy_score(ens_gt, ens_preds), 4),
    'F1_Weighted': round(f1_score(ens_gt, ens_preds, average='weighted'), 4),
    'F1_Macro': round(f1_score(ens_gt, ens_preds, average='macro'), 4),
    'F1_Normal': round(f1_score(ens_gt, ens_preds, labels=[0], average='macro'), 4),
    'F1_Benign': round(f1_score(ens_gt, ens_preds, labels=[1], average='macro'), 4),
    'F1_Malignant': round(f1_score(ens_gt, ens_preds, labels=[2], average='macro'), 4),
    'Precision_Weighted': round(precision_score(ens_gt, ens_preds, average='weighted', zero_division=0), 4),
    'Precision_Normal': round(precision_score(ens_gt, ens_preds, labels=[0], average='macro', zero_division=0), 4),
    'Precision_Benign': round(precision_score(ens_gt, ens_preds, labels=[1], average='macro', zero_division=0), 4),
    'Precision_Malignant': round(precision_score(ens_gt, ens_preds, labels=[2], average='macro', zero_division=0), 4),
    'Recall_Weighted': round(recall_score(ens_gt, ens_preds, average='weighted', zero_division=0), 4),
    'Recall_Normal': round(recall_score(ens_gt, ens_preds, labels=[0], average='macro', zero_division=0), 4),
    'Recall_Benign': round(recall_score(ens_gt, ens_preds, labels=[1], average='macro', zero_division=0), 4),
    'Recall_Malignant': round(recall_score(ens_gt, ens_preds, labels=[2], average='macro', zero_division=0), 4),
}

# ── QSVM Overall Metrics (if available) ──
rows = [mtunet_overall, ens_overall]
if all_qsvm_preds:
    qsvm_preds_all = np.concatenate(all_qsvm_preds)
    qsvm_gt_all    = np.concatenate(all_qsvm_gt)
    qsvm_overall = {
        'Method': 'MTUNetPlusPlus + QSVM',
        'Accuracy': round(accuracy_score(qsvm_gt_all, qsvm_preds_all), 4),
        'F1_Weighted': round(f1_score(qsvm_gt_all, qsvm_preds_all, average='weighted'), 4),
        'F1_Macro': round(f1_score(qsvm_gt_all, qsvm_preds_all, average='macro'), 4),
        'F1_Normal': round(f1_score(qsvm_gt_all, qsvm_preds_all, labels=[0], average='macro'), 4),
        'F1_Benign': round(f1_score(qsvm_gt_all, qsvm_preds_all, labels=[1], average='macro'), 4),
        'F1_Malignant': round(f1_score(qsvm_gt_all, qsvm_preds_all, labels=[2], average='macro'), 4),
        'Precision_Weighted': round(precision_score(qsvm_gt_all, qsvm_preds_all, average='weighted', zero_division=0), 4),
        'Precision_Normal': round(precision_score(qsvm_gt_all, qsvm_preds_all, labels=[0], average='macro', zero_division=0), 4),
        'Precision_Benign': round(precision_score(qsvm_gt_all, qsvm_preds_all, labels=[1], average='macro', zero_division=0), 4),
        'Precision_Malignant': round(precision_score(qsvm_gt_all, qsvm_preds_all, labels=[2], average='macro', zero_division=0), 4),
        'Recall_Weighted': round(recall_score(qsvm_gt_all, qsvm_preds_all, average='weighted', zero_division=0), 4),
        'Recall_Normal': round(recall_score(qsvm_gt_all, qsvm_preds_all, labels=[0], average='macro', zero_division=0), 4),
        'Recall_Benign': round(recall_score(qsvm_gt_all, qsvm_preds_all, labels=[1], average='macro', zero_division=0), 4),
        'Recall_Malignant': round(recall_score(qsvm_gt_all, qsvm_preds_all, labels=[2], average='macro', zero_division=0), 4),
    }
    rows.append(qsvm_overall)

# Save to Excel
final_df = pd.DataFrame(rows)
excel_path = Path(RUN_PATH) / "FINAL_OVERALL_RESULTS.xlsx"
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    final_df.to_excel(writer, sheet_name='Overall_Metrics', index=False)

    # Also add paper comparison
    paper_row = {
        'Method': 'Paper (MTUNet++ MT+PR+DO)',
        'Accuracy': 0.802,
        'F1_Weighted': 0.801,
        'F1_Macro': None,
        'F1_Normal': 0.741,
        'F1_Benign': 0.826,
        'F1_Malignant': 0.791,
        'Precision_Weighted': None,
        'Precision_Normal': None,
        'Precision_Benign': None,
        'Precision_Malignant': None,
        'Recall_Weighted': None,
        'Recall_Normal': None,
        'Recall_Benign': None,
        'Recall_Malignant': None,
    }
    comparison_df = pd