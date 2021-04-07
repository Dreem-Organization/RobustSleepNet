import pandas as pd
import matplotlib
from matplotlib.ticker import (MultipleLocator)
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

df = pd.read_csv('scripts/plots/fig_3/DT_results.csv')
df_supervised = pd.read_csv('scripts/plots/fig_3/LfS_results.csv')

df['dataset'] = [x.replace('_', ' ').upper() for x in df['dataset']]
datasets = df['dataset'].unique()
df_supervised['dataset'] = [x.replace('_', ' ').upper()
                            if x != 'sleep_edf_extended' else 'SLEEP EDF' for x in
                            df_supervised['dataset']]

df = df[['dataset', 'training_set_size', 'f1_macro']]

df_supervised = df_supervised[['dataset', 'f1_macro']]
df_supervised.columns = ['dataset', 'supervised_learning_performance']
df = df.merge(df_supervised, left_on='dataset', right_on='dataset')
df['f1_macro_true'] = df['f1_macro']
df['f1_macro'] = df['f1_macro'] / df['supervised_learning_performance'] * 100

df_groupped = df.groupby(['dataset', 'training_set_size']).mean()
df_groupped['max'] = df.groupby(['dataset', 'training_set_size']).max()['f1_macro']
df_groupped['min'] = df.groupby(['dataset', 'training_set_size']).min()['f1_macro']
df_groupped['count'] = df.groupby(['dataset', 'training_set_size']).count()['f1_macro']
df_groupped = df_groupped.reset_index()

params = {
    'legend.markerscale': 2,
    'font.size': 30,
    'font.family': 'Open Sans',
}
plt.rcParams.update(params)
f, axis = plt.subplots(len(datasets), 1, sharex=True, sharey=True, figsize=(14, 20))
plt.subplots_adjust(left=0.10, right=0.95, top=0.98, bottom=0.05, hspace=0.05)
datasets_list = ['DODO', 'DODH', 'MASS', 'SLEEP EDF', "MROS", "MESA", "SHHS", "CAP"]
for i, dataset in enumerate(datasets_list):
    data = df_groupped[df_groupped["dataset"] == dataset]
    axis[i].plot(data['training_set_size'], data['f1_macro'], c="#ED5C79", linewidth=4)
    axis[i].set_ylabel(dataset, labelpad=10)
    axis[i].spines['right'].set_visible(False)
    axis[i].spines['top'].set_visible(False)
    axis[i].hlines(100, 0, 1000, linewidth=2, linestyles='dotted')
    if i < len(datasets) - 1:
        axis[i].set_xticks([])
    else:
        axis[i].set_xticks([2 * 7, 8 * 7, 16 * 7, 32 * 7, 64 * 7, 128 * 7])
    axis[i].yaxis.set_major_locator(MultipleLocator(20))

    # For the minor ticks, use no labels; default NullFormatter.
    axis[i].yaxis.set_minor_locator(MultipleLocator(5))

    for k, elt in data.iterrows():
        axis[i].arrow(elt['training_set_size'], elt['f1_macro'], 0, elt['min'] - elt['f1_macro'],
                      length_includes_head=True,
                      head_width=5, head_length=0.1, zorder=4)
        axis[i].arrow(elt['training_set_size'], elt['f1_macro'], 0, elt['max'] - elt['f1_macro'],
                      length_includes_head=True,
                      head_width=5, head_length=0.1, zorder=4)

plt.xlim(0, 1000)
plt.xlabel('Number of pretraining records')
