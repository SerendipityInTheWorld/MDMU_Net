import numpy as np
from scipy.stats import ttest_rel, shapiro

our_pc = []  # our model
Net = [] # else model

differences = np.array(our_pc) - np.array(Net)

_, p_shapiro = shapiro(differences)
print(f"Shapiro p-value: {p_shapiro}")

if p_shapiro > 0.05:
    _, p_value = ttest_rel(our_pc, Net)
    print(f"t-test p-value: {p_value}")
else:
    # Wilcoxon
    from scipy.stats import wilcoxon
    _, p_value = wilcoxon(differences)
    print(f"Wilcoxon p-value: {p_value}")