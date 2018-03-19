import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('../results/reports/report_seq.csv', sep='\t', encoding='utf-8')

df2 = pd.read_csv('../results/reports/report_fm.csv', sep='\t', encoding='utf-8')

df3 = pd.concat([df1, df2])

models = df3['model'].values

score = df3['f1_macro'].values

fig, ax = plt.subplots()

index = np.arange(len(models))

bar_width = 0.4

opacity = 0.4

rects1 = plt.bar(index, score, bar_width,
                 alpha=opacity)

plt.xlabel('Model')
plt.ylabel('F1 Macro')
plt.title('Testing Set F1 Macro by Model')
plt.xticks(index, models, rotation=60, ha='right')

plt.tight_layout()
plt.show()
#plt.savefig('../report_figures/test_f1macro.png')
