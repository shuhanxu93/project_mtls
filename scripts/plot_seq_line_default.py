import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]

df = pd.read_csv('../results/window_none.csv', sep='\t', encoding='utf-8')
datasets = df.loc[:,['dataset_0', 'dataset_1', 'dataset_2', 'dataset_3', 'dataset_4']].as_matrix()
y1_mean = np.mean(datasets, axis=1).tolist()
y1_std = np.std(datasets, axis=1).tolist()

df = pd.read_csv('../results/window_balanced.csv', sep='\t', encoding='utf-8')
datasets = df.loc[:,['dataset_0', 'dataset_1', 'dataset_2', 'dataset_3', 'dataset_4']].as_matrix()
y2_mean = np.mean(datasets, axis=1).tolist()
y2_std = np.std(datasets, axis=1).tolist()

fig, ax = plt.subplots()

line1 = ax.errorbar(x, y1_mean, yerr=y1_std, capsize=4, label='class_weight=None')
line2 = ax.errorbar(x, y2_mean, yerr=y2_std, capsize=4, label='class_weight=\'balanced\'')
ax.set_xticks(x)
ax.set_ylim(0.30, 0.7)
ax.set_title("Five-fold Cross Validation Accuracy using Default Hyper-parameters, SVC(kernel='rbf')")
ax.set_xlabel("Window Size")
ax.set_ylabel("Accuracy")
ax.legend(loc='lower right')
plt.show()
