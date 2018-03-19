import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# testing set
df1 = pd.read_csv('../results/reports/report_seq.csv', sep='\t', encoding='utf-8')

df2 = pd.read_csv('../results/reports/report_fm.csv', sep='\t', encoding='utf-8')

df_test = pd.concat([df1, df2])

# new proteins
df3 = pd.read_csv('../results/reports/report_seq_newproteins.csv', sep='\t', encoding='utf-8')

df4 = pd.read_csv('../results/reports/report_fm_newproteins.csv', sep='\t', encoding='utf-8')

df_newproteins = pd.concat([df3, df4])


models = df_test['model'].values

score_test = df_test['f1_macro'].values

score_newproteins = df_newproteins['f1_macro'].values


fig, ax = plt.subplots()

index = np.arange(len(models))

bar_width = 0.3

opacity = 0.4

rects1 = plt.bar(index, score_test, bar_width,
                 alpha=opacity, label='Testing Set')

rects2 = plt.bar(index + bar_width, score_newproteins, bar_width,
                 alpha=opacity, label='New Proteins')

plt.xlabel('Model')
plt.ylabel('F1 Macro')
plt.title('F1 Macro by Model (Testing Set and New Proteins)')
plt.xticks(index + bar_width / 2, models, rotation=60, ha='right')
plt.legend(loc='upper left')
plt.ylim(0, 1)

plt.tight_layout()
plt.show()
#plt.savefig('../report_figures/f1macro.png')
