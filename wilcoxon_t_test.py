import numpy as np
from scipy.stats import ttest_rel, wilcoxon

a = np.array([0.5056, 0.5681, 0.5381, 0.5248, 0.4764])
b = np.array([0.4375, 0.5521, 0.4953, 0.5262, 0.4921])

t_stat, p_ttest = ttest_rel(a, b)
print(f"Paired t-test: t={t_stat:.3f}, p={p_ttest:.3f}")

w_stat, p_wilcoxon = wilcoxon(a, b)
print(f"Wilcoxon: W={w_stat}, p={p_wilcoxon:.3f}")