import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import pandas as pd

df = pd.read_csv('../results/seq_rbf/seq_rbf_none_15.csv', sep='\t', encoding='utf-8')

df2 = pd.pivot_table(df, values='mean_test_score', index='param_gamma', columns='param_C')

print(df2)


x = df2.columns.values
y = df2.index.values
X, Y = np.meshgrid(x, y)
Z = df2.values
levels = [0.60, 0.605, 0.61, 0.615, 0.62, 0.625, 0.63, 0.635, 0.64, 0.645, 0.65, 0.655, 0.66]

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
