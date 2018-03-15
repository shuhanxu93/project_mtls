import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import pandas as pd

df = pd.read_csv('../results/pssm_svm_rbf/fm_rbf_balanced_15.csv', sep='\t', encoding='utf-8')

df2 = pd.pivot_table(df, values='mean_test_accuracy', index='param_gamma', columns='param_C')

print(df2)


x = df2.columns.values
y = df2.index.values
X, Y = np.meshgrid(x, y)
Z = df2.values
levels = [0.700, 0.705, 0.710, 0.715, 0.720, 0.725, 0.730, 0.735, 0.740, 0.745,
          0.750, 0.755, 0.760]

print(Z.max())

fig, ax = plt.subplots()
plt.title("Five-fold Cross Validation Accuracy (RBF, class_weight=None)")
plt.xlabel("C")
plt.ylabel("gamma")
ax.set_xscale('log', basex=2)
ax.set_yscale('log', basey=2)
cs = ax.contourf(X, Y, Z, levels, cmap=cm.hot)
cbar = fig.colorbar(cs)


plt.show()
