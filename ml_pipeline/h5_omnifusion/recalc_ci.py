import math

def wilson_ci(k, n, confidence=0.95):
    if n == 0: return 0.0, 0.0
    z = 1.96
    p = k / n
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    error = z * math.sqrt((p*(1-p)/n + z**2/(4*n**2))) / denom
    return (center - error, center + error)

TP = 47
FN = 4
TN = 14
FP = 10

n_pos = TP + FN
n_neg = TN + FP
total = TP + FN + TN + FP

sens = TP / n_pos
spec = TN / n_neg
acc = (TP + TN) / total
ppv = TP / (TP + FP)
npv = TN / (TN + FN)

sens_ci = wilson_ci(TP, n_pos)
spec_ci = wilson_ci(TN, n_neg)
acc_ci = wilson_ci(TP + TN, total)

print(f"Sensitivity (Recall): {sens:.4%} [{sens_ci[0]:.4%}, {sens_ci[1]:.4%}]")
print(f"Specificity:          {spec:.4%} [{spec_ci[0]:.4%}, {spec_ci[1]:.4%}]")
print(f"Accuracy:             {acc:.4%} [{acc_ci[0]:.4%}, {acc_ci[1]:.4%}]")
print(f"PPV (Precision):      {ppv:.4%}")
print(f"NPV:                  {npv:.4%}")
