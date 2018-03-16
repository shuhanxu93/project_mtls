import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import pandas as pd

levels = [0.60, 0.605, 0.61, 0.615, 0.62, 0.625, 0.63, 0.635, 0.64, 0.645, 0.65, 0.655, 0.66]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
fig.delaxes(axes[2, 1])
fig.delaxes(axes[2, 2])
fs = 10

df = pd.read_csv('../results/seq_rbf/seq_rbf_none_11.csv', sep='\t', encoding='utf-8')
df2 = pd.pivot_table(df, values='mean_test_score', index='param_gamma', columns='param_C')
x = df2.columns.values
y = df2.index.values
X, Y = np.meshgrid(x, y)
Z = df2.values
axes[0, 0].contourf(X, Y, Z, levels, cmap=cm.hot)
axes[0, 0].set_title('Window Size 11', fontsize=fs)

df = pd.read_csv('../results/seq_rbf/seq_rbf_none_13.csv', sep='\t', encoding='utf-8')
df2 = pd.pivot_table(df, values='mean_test_score', index='param_gamma', columns='param_C')
x = df2.columns.values
y = df2.index.values
X, Y = np.meshgrid(x, y)
Z = df2.values
axes[0, 1].contourf(X, Y, Z, levels, cmap=cm.hot)
axes[0, 1].set_title('Window Size 13', fontsize=fs)

df = pd.read_csv('../results/seq_rbf/seq_rbf_none_15.csv', sep='\t', encoding='utf-8')
df2 = pd.pivot_table(df, values='mean_test_score', index='param_gamma', columns='param_C')
x = df2.columns.values
y = df2.index.values
X, Y = np.meshgrid(x, y)
Z = df2.values
axes[0, 2].contourf(X, Y, Z, levels, cmap=cm.hot)
axes[0, 2].set_title('Window Size 15', fontsize=fs)

df = pd.read_csv('../results/seq_rbf/seq_rbf_none_17.csv', sep='\t', encoding='utf-8')
df2 = pd.pivot_table(df, values='mean_test_score', index='param_gamma', columns='param_C')
x = df2.columns.values
y = df2.index.values
X, Y = np.meshgrid(x, y)
Z = df2.values
axes[1, 0].contourf(X, Y, Z, levels, cmap=cm.hot)
axes[1, 0].set_title('Window Size 17', fontsize=fs)

df = pd.read_csv('../results/seq_rbf/seq_rbf_none_19.csv', sep='\t', encoding='utf-8')
df2 = pd.pivot_table(df, values='mean_test_score', index='param_gamma', columns='param_C')
x = df2.columns.values
y = df2.index.values
X, Y = np.meshgrid(x, y)
Z = df2.values
axes[1, 1].contourf(X, Y, Z, levels, cmap=cm.hot)
axes[1, 1].set_title('Window Size 19', fontsize=fs)

df = pd.read_csv('../results/seq_rbf/seq_rbf_none_21.csv', sep='\t', encoding='utf-8')
df2 = pd.pivot_table(df, values='mean_test_score', index='param_gamma', columns='param_C')
x = df2.columns.values
y = df2.index.values
X, Y = np.meshgrid(x, y)
Z = df2.values
axes[1, 2].contourf(X, Y, Z, levels, cmap=cm.hot)
axes[1, 2].set_title('Window Size 21', fontsize=fs)

df = pd.read_csv('../results/seq_rbf/seq_rbf_none_23.csv', sep='\t', encoding='utf-8')
df2 = pd.pivot_table(df, values='mean_test_score', index='param_gamma', columns='param_C')
x = df2.columns.values
y = df2.index.values
X, Y = np.meshgrid(x, y)
Z = df2.values
cs = axes[2, 0].contourf(X, Y, Z, levels, cmap=cm.hot)
axes[2, 0].set_title('Window Size 23', fontsize=fs)


for ax in axes.flatten():
    ax.set_xlabel('C')
    ax.set_ylabel('gamma')
    ax.set_xscale('log', basex=2)
    ax.set_yscale('log', basey=2)

fig.suptitle("Five-fold Cross Validation Accuracy (SVC(rbf), class_weight=None)", fontsize=20)
fig.tight_layout()
fig.subplots_adjust(top=0.93, wspace=0.25, hspace=0.25)
cbar_ax = fig.add_axes([0.35, 0.043, 0.025, 0.253])
cbar = fig.colorbar(cs, cax=cbar_ax)
plt.savefig('here.png')
