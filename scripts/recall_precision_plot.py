import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# new proteins
df3 = pd.read_csv('../results/reports/report_seq_newproteins.csv', sep='\t', encoding='utf-8')

df4 = pd.read_csv('../results/reports/report_fm_newproteins.csv', sep='\t', encoding='utf-8')

df_newproteins = pd.concat([df3, df4]).reset_index(drop=True)


models = df_newproteins.loc[[0, 9], 'model'].values

score_H = df_newproteins.loc[[0, 9], 'precision_H'].values

score_E = df_newproteins.loc[[0, 9], 'precision_E'].values

score_C = df_newproteins.loc[[0, 9], 'precision_C'].values


fig, ax = plt.subplots()

index = np.arange(len(models))

bar_width = 0.25

opacity = 0.4

rects1 = plt.bar(index, score_H, bar_width,
                 alpha=opacity, label='H')

rects2 = plt.bar(index + bar_width, score_E, bar_width,
                 alpha=opacity, label='E')

rects3 = plt.bar(index + 2 * bar_width, score_C, bar_width,
                 alpha=opacity, label='C')

plt.xlabel('Model')
plt.ylabel('Precision')
plt.title('Precision per class by Model (New Proteins)')
plt.xticks(index + 2 * bar_width / 2, models, rotation=60, ha='right')
plt.legend(loc='upper left')
plt.ylim(0, 1)

plt.tight_layout()
#plt.show()
plt.savefig('../report_figures/bestmodels_precision.png')
