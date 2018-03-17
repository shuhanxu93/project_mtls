import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import pandas as pd


x = np.power(2, np.linspace(-5, 15, 11))
y = np.array([11, 13, 15, 17, 19, 21, 23])
z = []
for window_size in [11, 13, 15, 17, 19, 21, 23]:
    filename = '../results/fm_linear/fm_linear_balanced_' + str(window_size) + '.csv'
    df = pd.read_csv(filename, sep='\t', encoding='utf-8')
    z.append(df['mean_test_score'].values)

X, Y = np.meshgrid(x, y)
Z = np.vstack(z)

print(Z)

levels = [0.700, 0.705, 0.710, 0.715, 0.720, 0.725, 0.730, 0.735, 0.740, 0.745,
          0.750, 0.755, 0.760]

fig, ax = plt.subplots()
plt.title("Five-fold Cross Validation Accuracy, LinearSVC(class_weight=\'balanced\')", fontsize=10)
plt.xlabel("C")
plt.ylabel("Window Size")
ax.set_xscale('log', basex=2)
cs = ax.contourf(X, Y, Z, levels, cmap=cm.hot)
cbar = fig.colorbar(cs)
#plt.show()
plt.savefig('../report_figures/contour_fm_linear_balanced.png')
