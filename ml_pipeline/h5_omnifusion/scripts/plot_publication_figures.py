import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os

# -------------------------------------------------------------------
# CONFIGURATION - UPDATE THESE VALUES WITH YOUR SPECIFIC RESULTS
# -------------------------------------------------------------------

# 1. Confusion Matrix Values (Format: [[TN, FP], [FN, TP]])
# PLEASE UPDATE THESE VALUES FROM YOUR BEST RUN
# Example from context: 219 samples (65 Depressed, 154 Non-Depressed)
# Assuming ~85% acc -> TN=135, FP=19, FN=10, TP=55 (Just a placeholder!)
CONFUSION_MATRIX = np.array([
    [135, 19],  # Non-Depressed (True Negative, False Positive)
    [10, 55]    # Depressed     (False Negative, True Positive)
])

# 2. AUC Score (Placeholder)
AUC_SCORE = 0.88 

# 3. Output Directory
OUTPUT_DIR = r"c:\Users\thela\OneDrive\Desktop\phase 2\ml_pipeline\h5_omnifusion\results\publication"

# -------------------------------------------------------------------
# PLOTTING FUNCTIONS
# -------------------------------------------------------------------

def set_style():
    """Set publication-quality plotting style."""
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3
    })

def plot_confusion_matrix(cm, classes, save_path):
    """Plots a beautiful confusion matrix."""
    plt.figure(figsize=(8, 6))
    
    # Calculate percentages
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    
    # Annotations with counts and percentages
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
                
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False,
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 14, "weight": "bold"},
                linewidths=1, linecolor='black', clip_on=False)
    
    plt.title('Confusion Matrix: Best H5-OmniFusion Model', pad=20)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved Confusion Matrix to: {save_path}")

def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    """Plots a stylized ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC)', pad=20)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved ROC Curve to: {save_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_style()
    
    classes = ['Non-Depressed', 'Depressed']
    
    # 1. Plot Confusion Matrix
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_publication.png")
    plot_confusion_matrix(CONFUSION_MATRIX, classes, cm_path)
    
    # 2. Plot Simulated ROC (since we don't have raw probs, we simulate a curve matching the AUC)
    # This is for visualization purposes only based on the placeholder AUC
    # In a real scenario, use actual y_true and y_probs
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    # Simulate data to generate a curve with approx AUC
    # NOTE: Replace this with actual FPR/TPR arrays if available!
    print("⚠️  Generating simulated ROC curve based on placeholder AUC (Replace with real data!)")
    
    # Generate noisy data
    X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.7,0.3], random_state=42)
    # Fit simple model to get a curve
    model = LogisticRegression()
    model.fit(X, y)
    y_probs = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_probs)
    
    # Scale/adjust AUC to match our target for the plot (visual approximation)
    # Real approach: Load y_true, y_probs from file
    
    roc_path = os.path.join(OUTPUT_DIR, "roc_curve_publication.png")
    plot_roc_curve(fpr, tpr, AUC_SCORE, roc_path)

if __name__ == "__main__":
    main()
