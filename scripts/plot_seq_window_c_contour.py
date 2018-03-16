import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import pandas as pd


x = np.power(2, np.linspace(-5, 15, 11))
y = np.array([11, 13, 15, 17, 19, 21, 23])
z = []
for window_size in [11, 13, 15, 17, 19, 21, 23]:
    filename = '../results/seq_linear/seq_linear_balanced_' + str(window_size) + '.csv'
    df = pd.read_csv(filename, sep='\t', encoding='utf-8')
    z.append(df['mean_test_score'].values)

X, Y = np.meshgrid(x, y)
Z = np.vstack(z)

print(Z)

levels = [0.60, 0.605, 0.61, 0.615, 0.62, 0.625, 0.63, 0.635, 0.64, 0.645, 0.65, 0.655, 0.66]

fig, ax = plt.subplots()
plt.title("5 fold cross-validation accuracy score for LinearSVC (class_weight=None)")
plt.xlabel("C")
plt.ylabel("Window Size")
ax.set_xscale('log', basex=2)
cs = ax.contourf(X, Y, Z, levels, cmap=cm.hot)
cbar = fig.colorbar(cs)


plt.show()
