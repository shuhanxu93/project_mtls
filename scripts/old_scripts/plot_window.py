import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../results/window_none.csv', sep='\t', encoding='utf-8')

plt.figure()
plt.title("Five-fold Cross Validation Accuracy using Default Hyper-parameters (SVC(rbf), class_weight=None)")
plt.xlabel("Window Size")
plt.ylabel("Accuracy")
window_sizes = df.loc[:,'window_size'].as_matrix()
datasets = df.loc[:,['dataset_0', 'dataset_1', 'dataset_2', 'dataset_3', 'dataset_4']].as_matrix()

print(datasets)

datasets_mean = np.mean(datasets, axis=1)
datasets_std = np.std(datasets, axis=1)


plt.grid()
plt.fill_between(window_sizes, datasets_mean - datasets_std,
                     datasets_mean + datasets_std, alpha=0.1,
                     color="r")

plt.plot(window_sizes, datasets_mean, 'o-', color="r",
             label="mean score")

plt.legend(loc="best")
plt.show()
