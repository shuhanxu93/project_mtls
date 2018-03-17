import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import pandas as pd

input_type = 'seq'
class_weight = 'none'

if input_type == 'fm':
    levels = [0.700, 0.705, 0.710, 0.715, 0.720, 0.725, 0.730, 0.735, 0.740, 0.745,
              0.750, 0.755, 0.760]
    metric_name = 'accuracy'
else:
    levels = [0.600, 0.605, 0.610, 0.615, 0.620, 0.625, 0.630, 0.635, 0.640, 0.645,
              0.65, 0.655, 0.66]
    metric_name = 'score'

if class_weight == 'balanced':
    class_weight_title = '\'balanced\''
else:
    class_weight_title = 'None'


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
fig.delaxes(axes[2, 1])
fig.delaxes(axes[2, 2])
fs = 10

df = pd.read_csv('../results/' + input_type + '_rbf/' + input_type + '_rbf_' + class_weight + '_11.csv', sep='\t', encoding='utf-8')
df2 = pd.pivot_table(df, values='mean_test_' + metric_name, index='param_gamma', columns='param_C')
x = df2.columns.values
y = df2.index.values
X, Y = np.meshgrid(x, y)
Z = df2.values
axes[0, 0].contourf(X, Y, Z, levels, cmap=cm.hot)
axes[0, 0].set_title('Window Size 11', fontsize=fs)

df = pd.read_csv('../results/' + input_type + '_rbf/' + input_type + '_rbf_' + class_weight + '_13.csv', sep='\t', encoding='utf-8')
df2 = pd.pivot_table(df, values='mean_test_' + metric_name, index='param_gamma', columns='param_C')
x = df2.columns.values
y = df2.index.values
X, Y = np.meshgrid(x, y)
Z = df2.values
axes[0, 1].contourf(X, Y, Z, levels, cmap=cm.hot)
axes[0, 1].set_title('Window Size 13', fontsize=fs)

df = pd.read_csv('../results/' + input_type + '_rbf/' + input_type + '_rbf_' + class_weight + '_15.csv', sep='\t', encoding='utf-8')
df2 = pd.pivot_table(df, values='mean_test_' + metric_name, index='param_gamma', columns='param_C')
x = df2.columns.values
y = df2.index.values
X, Y = np.meshgrid(x, y)
Z = df2.values
axes[0, 2].contourf(X, Y, Z, levels, cmap=cm.hot)
axes[0, 2].set_title('Window Size 15', fontsize=fs)

df = pd.read_csv('../results/' + input_type + '_rbf/' + input_type + '_rbf_' + class_weight + '_17.csv', sep='\t', encoding='utf-8')
df2 = pd.pivot_table(df, values='mean_test_' + metric_name, index='param_gamma', columns='param_C')
x = df2.columns.values
y = df2.index.values
X, Y = np.meshgrid(x, y)
Z = df2.values
axes[1, 0].contourf(X, Y, Z, levels, cmap=cm.hot)
axes[1, 0].set_title('Window Size 17', fontsize=fs)

df = pd.read_csv('../results/' + input_type + '_rbf/' + input_type + '_rbf_' + class_weight + '_19.csv', sep='\t', encoding='utf-8')
df2 = pd.pivot_table(df, values='mean_test_' + metric_name, index='param_gamma', columns='param_C')
x = df2.columns.values
y = df2.index.values
X, Y = np.meshgrid(x, y)
Z = df2.values
axes[1, 1].contourf(X, Y, Z, levels, cmap=cm.hot)
axes[1, 1].set_title('Window Size 19', fontsize=fs)

df = pd.read_csv('../results/' + input_type + '_rbf/' + input_type + '_rbf_' + class_weight + '_21.csv', sep='\t', encoding='utf-8')
df2 = pd.pivot_table(df, values='mean_test_' + metric_name, index='param_gamma', columns='param_C')
x = df2.columns.values
y = df2.index.values
X, Y = np.meshgrid(x, y)
Z = df2.values
axes[1, 2].contourf(X, Y, Z, levels, cmap=cm.hot)
axes[1, 2].set_title('Window Size 21', fontsize=fs)

df = pd.read_csv('../results/' + input_type + '_rbf/' + input_type + '_rbf_' + class_weight + '_23.csv', sep='\t', encoding='utf-8')
df2 = pd.pivot_table(df, values='mean_test_' + metric_name, index='param_gamma', columns='param_C')
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

fig.suptitle("Five-fold Cross Validation Accuracy, SVC(kernel='rbf', class_weight=" + class_weight_title + ")", fontsize=20)
fig.tight_layout()
fig.subplots_adjust(top=0.93, wspace=0.25, hspace=0.25)
cbar_ax = fig.add_axes([0.35, 0.043, 0.025, 0.253])
cbar = fig.colorbar(cs, cax=cbar_ax)
plt.savefig('../report_figures/contour_' + input_type + '_rbf_' + class_weight + '.png')
