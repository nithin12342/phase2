import json
import os

nb = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "header"
   },
   "source": [
    "# 📊 Best Model Evaluation: Threshold Analysis\n",
    "Evaluates the **single best model** (`h5_omnifusion_medium_fold4_best.pt`) and plots metrics across decision thresholds."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {
    "id": "setup"
   },
   "outputs": [],
   "source": [
    "#Setup Environment\n",
    "from google.colab import drive\n",
    "import os, sys, subprocess\n",
    "\n",
    "drive.mount('/content/drive')\n",
    "\n",
    "if not os.path.exists('/content/phase2'):\n",
    "    subprocess.run(['git', 'clone', 'https://github.com/nithin12342/phase2.git', '/content/phase2'])\n",
    "else:\n",
    "    subprocess.run(['git', '-C', '/content/phase2', 'pull'])\n",
    "\n",
    "PROJECT_DIR = '/content/phase2/ml_pipeline/h5_omnifusion'\n",
    "os.chdir(PROJECT_DIR)\n",
    "if PROJECT_DIR not in sys.path:\n",
    "    sys.path.insert(0, PROJECT_DIR)\n",
    "\n",
    "subprocess.run(['pip', 'install', 'torch', 'torchvision', 'torchaudio', 'h5py',\n",
    "                'pandas', 'scikit-learn', 'matplotlib', 'seaborn', '--quiet'])\n",
    "\n",
    "print('✅ Environment Ready')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {
    "id": "evaluate"
   },
   "outputs": [],
   "source": [
    "#Evaluate Single Best Model\n",
    "import torch\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix\n",
    "from collections import OrderedDict\n",
    "from config.model_config import H5Config, ComputeTier\n",
    "from src.models.h5_omnifusion import H5OmniFusion\n",
    "from src.data.h5_dataset import create_h5_dataloaders_kfold\n",
    "from tqdm import tqdm\n",
    "\n",
    "MODEL_PATH = '/content/drive/MyDrive/DAIC-WOZ_Datasets/checkpoints_phase9/h5_omnifusion_medium_fold4_best.pt'\n",
    "DATA_DIR = '/content/drive/MyDrive/DAIC-WOZ_Datasets/H5_OmniFusion_Output'\n",
    "LABELS_CSV = f'{DATA_DIR}/all_labels.csv'\n",
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "\n",
    "def to_device(data, device):\n",
    "    if isinstance(data, torch.Tensor): return data.to(device)\n",
    "    elif isinstance(data, dict): return {k: to_device(v, device) for k, v in data.items()}\n",
    "    return data\n",
    "\n",
    "# Load Data\n",
    "_, _, test_loader = create_h5_dataloaders_kfold(\n",
    "    h5_dir=DATA_DIR, labels_csv=LABELS_CSV,\n",
    "    batch_size=32, fold_idx=4, n_folds=5  # Using Fold 4 test set\n",
    ")\n",
    "\n",
    "# Load Model\n",
    "print(f'🔧 Loading {MODEL_PATH}...')\n",
    "checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)\n",
    "sd = checkpoint.get('model_state_dict', checkpoint)\n",
    "sd = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in sd.items())\n",
    "\n",
    "config = H5Config.from_tier(ComputeTier.MEDIUM)\n",
    "model = H5OmniFusion(config)\n",
    "model.load_state_dict(sd, strict=False)\n",
    "model.to(DEVICE)\n",
    "model.eval()\n",
    "print('✅ Model Loaded')\n",
    "\n",
    "# Get Predictions\n",
    "all_labels, all_probs = [], []\n",
    "with torch.no_grad():\n",
    "    for batch in tqdm(test_loader, desc='Evaluating'):\n",
    "        input_keys = [k for k in batch.keys() if k not in ['label', 'labels', 'target', 'targets', 'participant_id']]\n",
    "        inputs = to_device({k: batch[k] for k in input_keys}, DEVICE)\n",
    "        labels = batch.get('label', batch.get('labels', {})).get('binary', torch.zeros(1)).to(DEVICE)\n",
    "        outputs = model(inputs)\n",
    "        all_probs.extend(outputs[0]['binary_prob'].cpu().numpy())\n",
    "        all_labels.extend(labels.cpu().numpy())\n",
    "\n",
    "y_true = np.array(all_labels)\n",
    "y_prob = np.array(all_probs)\n",
    "print(f'✅ Evaluated {len(y_true)} samples')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {
    "id": "plot"
   },
   "outputs": [],
   "source": [
    "#Plot Metrics vs Threshold Line Graph\n",
    "thresholds = np.arange(0.30, 0.85, 0.05)\n",
    "f1s, precs, recs, accs = [], [], [], []\n",
    "\n",
    "for t in thresholds:\n",
    "    y_pred = (y_prob >= t).astype(int)\n",
    "    f1s.append(f1_score(y_true, y_pred, zero_division=0))\n",
    "    precs.append(precision_score(y_true, y_pred, zero_division=0))\n",
    "    recs.append(recall_score(y_true, y_pred, zero_division=0))\n",
    "    accs.append(accuracy_score(y_true, y_pred))\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.plot(thresholds, f1s, 'b-o', label='F1 Score', linewidth=2)\n",
    "plt.plot(thresholds, precs, 'g--s', label='Precision', linewidth=2)\n",
    "plt.plot(thresholds, recs, 'r-.^', label='Recall', linewidth=2)\n",
    "plt.plot(thresholds, accs, 'm:d', label='Accuracy', linewidth=2)\n",
    "\n",
    "plt.title('Performance Metrics vs Decision Threshold (Best Model)', fontsize=14, fontweight='bold')\n",
    "plt.xlabel('Decision Threshold', fontsize=12)\n",
    "plt.ylabel('Score', fontsize=12)\n",
    "plt.legend(fontsize=11)\n",
    "plt.grid(True, alpha=0.3)\n",
    "plt.ylim(0.5, 1.05)\n",
    "\n",
    "# Mark default 0.5\n",
    "plt.axvline(0.5, color='gray', linestyle=':', alpha=0.7)\n",
    "plt.text(0.51, 0.52, 'Default (0.5)', color='gray', fontsize=9)\n",
    "\n",
    "save_path = '/content/drive/MyDrive/DAIC-WOZ_Datasets/achieved/threshold_analysis.png'\n",
    "plt.savefig(save_path, dpi=150)\n",
    "plt.show()\n",
    "print(f'✅ Graph Saved: {save_path}')"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "gpuType": "T4",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}

with open("c:\\Users\\thela\\OneDrive\\Desktop\\phase 2\\ml_pipeline\\h5_omnifusion\\notebooks\\Best_Model_Evaluation.ipynb", "w") as f:
    json.dump(nb, f, indent=4)
print("✅ Notebook updated: Best_Model_Evaluation.ipynb")
