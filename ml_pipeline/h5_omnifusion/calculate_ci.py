import numpy as np
import math

def wilson_score_interval(k, n, confidence=0.95):
    if n == 0: return 0, 0
    z = 1.96 # Approx for 95%
    p = k / n
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    error = z * math.sqrt((p*(1-p)/n + z**2/(4*n**2))) / denom
    return (center - error, center + error)

TP = 50
FN = 1
TN = 13
FP = 11

sens = TP / (TP + FN)
spec = TN / (TN + FP)
acc = (TP + TN) / (TP + TN + FP + FN)

sens_ci = wilson_score_interval(TP, TP + FN)
spec_ci = wilson_score_interval(TN, TN + FP)
acc_ci = wilson_score_interval(TP + TN, TP + TN + FP + FN)

print(f"Sensitivity: {sens:.4%} CI: [{sens_ci[0]:.4%}, {sens_ci[1]:.4%}]")
print(f"Specificity: {spec:.4%} CI: [{spec_ci[0]:.4%}, {spec_ci[1]:.4%}]")
print(f"Accuracy:    {acc:.4%} CI: [{acc_ci[0]:.4%}, {acc_ci[1]:.4%}]")

n_boot = 10000
np.random.seed(42)
y_true = np.array([1]*51 + [0]*24)
y_pred = np.array([1]*50 + [0]*1 + [1]*11 + [0]*13) # TP=50, FN=1, FP=11, TN=13

boot_f1s = []
for _ in range(n_boot):
    indices = np.random.randint(0, len(y_true), len(y_true))
    yt = y_true[indices]
    yp = y_pred[indices]
    
    tp = np.sum((yt == 1) & (yp == 1))
    fp = np.sum((yt == 0) & (yp == 1))
    fn = np.sum((yt == 1) & (yp == 0))
    
    f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0
    boot_f1s.append(f1)

boot_ci = np.percentile(boot_f1s, [2.5, 97.5])
print(f"Bootstrap F1 CI: [{boot_ci[0]:.4f}, {boot_ci[1]:.4f}]")
