import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = [11, 13, 15, 17, 19, 21, 23]
y1_mean = []
y1_std = []
y2_mean = []
y2_std = []

for window_size in x:
    filename = '../results/pssm_svm_rbf/fm_rbf_none_' + str(window_size) + '.csv'
    df = pd.read_csv(filename, sep='\t', encoding='utf-8')
    y1_mean.append(df['mean_test_accuracy'].values.max())
    y1_std.append(df['std_test_accuracy'].values[df['mean_test_accuracy'].values.argmax()])

for window_size in x:
    filename = '../results/pssm_svm_rbf/fm_rbf_balanced_' + str(window_size) + '.csv'
    df = pd.read_csv(filename, sep='\t', encoding='utf-8')
    y2_mean.append(df['mean_test_accuracy'].values.max())
    y2_std.append(df['std_test_accuracy'].values[df['mean_test_accuracy'].values.argmax()])

fig, ax = plt.subplots()

line1 = ax.errorbar(x, y1_mean, yerr=y1_std, capsize=4, label='class_weight=None')
line2 = ax.errorbar(x, y2_mean, yerr=y2_std, capsize=4, label='class_weight=\'balanced\'')
ax.set_xticks(x)
ax.set_ylim(0.65, 0.8)
ax.set_title("Five-fold Cross Validation Accuracy (SVC(RBF))")
ax.set_xlabel("Window Size")
ax.set_ylabel("Accuracy Score")
ax.legend(loc='lower right')
plt.show()
