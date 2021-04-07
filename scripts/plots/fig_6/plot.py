import pandas as pd
import matplotlib

import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('scripts/plots/learning_curve_finetuning/FT_learning_curve.csv')
df_supervised = pd.read_csv('scripts/plots/learning_curve_finetuning/LfS_results.csv')
df_finetuning = pd.read_csv('scripts/plots/learning_curve_finetuning/FT_results.csv')

df['dataset'] = [x.replace('_', ' ').upper() for x in df['dataset']]
df_finetuning['dataset'] = [x.replace('_', ' ').upper() for x in df_finetuning['dataset']]
df_supervised['dataset'] = [x.replace('_', ' ').upper()
                            if x != 'sleep_edf_extended' else 'SLEEP EDF' for x in
                            df_supervised['dataset']]

df = df[['dataset', 'n_records', 'f1_macro', 'records']]
df = df[df['n_records'] <= df['records'] * 4 / 5]
df_supervised = df_supervised[['dataset', 'f1_macro']]
df_supervised.columns = ['dataset', 'supervised_learning_performance']
df_finetuning = df_finetuning[['dataset', 'f1_macro']]
df_finetuning.columns = ['dataset', 'finetuning_performance']
df = df.merge(df_supervised, left_on='dataset', right_on='dataset')
df = df.merge(df_finetuning, left_on='dataset', right_on='dataset')
df['f1_macro'] = df['f1_macro'] / df['supervised_learning_performance'] * 100
df['finetuning_performance'] = df['finetuning_performance'] / df['supervised_learning_performance'] * 100
df = df.sort_values('dataset')

df_ = df[['dataset', 'f1_macro', "records", "n_records", "finetuning_performance"]]
df_ = df_.sort_values("n_records", ascending=False)[['dataset', 'f1_macro', "records", "finetuning_performance"]]
df_dt = df_.drop_duplicates(keep="last", subset=["dataset"]).sort_values('dataset')
df_ = df_.drop_duplicates(keep="first", subset=["dataset"]).sort_values('dataset')
colors = []

params = {'legend.fontsize': 25,
          'legend.markerscale': 3,
          'font.size': 28,
          'axes.titlesize': 24,
          'xtick.labelsize': 24,
          'ytick.labelsize': 24,
          'axes.labelsize': 24,
          'font.family': 'Open Sans',
          'axes.facecolor': 'white'
          }
sns.set(rc=params)
g = sns.lineplot('n_records', 'f1_macro', hue='dataset', data=df)
for l in g.lines:
    l.set_linewidth(3)
    l.set_linestyle('--')
    colors += [l._color]

# sc = plt.scatter(np.minimum(df_["records"] * 4/5,100),df_['finetuning_performance'] ,marker = "+",s = 200,c=colors,label = 'FT')
sc_dt = plt.scatter(df_dt["records"] * 0, df_dt['f1_macro'], marker="x", s=200, c=colors, label='DT')
plt.hlines(100, -1, 100)

# g = g.map(plt.scatter, 'training_set_size', 'f1_macro', color="#ED5C79")
g.set(xlabel="Number of Finetuning records",
      ylabel="% of the LFS F1",
      xticks=[0, 10, 20, 40, 60, 80, 100],
      xticklabels=[0, 10, 20, 40, 60, 80, 100])

plt.xlim(-1, 80)
# where some data has already been plotted to ax
handles, labels = g.get_legend_handles_labels()

import matplotlib.lines as mlines

labels[0] = 'Datasets'
handles, labels = handles[:-2], labels[:-2]
handles.append(mlines.Line2D([], [], color='black', marker='None', linestyle='None',
                             markersize=6))
labels.append('Baseline:')
labels += ['Direct Transfer']
handles += [mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                          markersize=6)]

# plt.hlines(100,35,1000)
# plt.set_axis_labels(, )
# g.set_titles(col_template="{col_name}")
plt.legend(handles, labels, loc='lower right', ncol=4)
# Hide the right and top spines
g._axes.spines['right'].set_visible(False)
g._axes.spines['top'].set_visible(False)
g._axes.spines['bottom'].set_visible(True)
g._axes.spines['left'].set_visible(True)
g.yaxis.set_ticks_position('left')
plt.show()
