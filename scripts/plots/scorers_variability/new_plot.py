import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
df = pd.read_csv('scripts/plots/scorers_variability/results.csv')
df.groupby(['scorer', 'source']).std()['f1_macro']

target_datasets = df['target'].unique()
target_datasets_order = {k: i for i, k in enumerate(target_datasets)}
source_datasets = ['dodo', 'dodh']
target_datasets = [x for x in target_datasets if x not in source_datasets]

for k, source_dataset in enumerate(source_datasets):
    df = pd.read_csv('scripts/plots/scorers_variability/results.csv')
    df = df[df['source'] == source_dataset]
    width = 0.4

    cmap = LinearSegmentedColormap.from_list("lol", ["#fff7fa", "#ffaaf6"])
    value = df[df['scorer'] != 'consensus']
    value['x'] = [target_datasets_order[x] for x in value['target']]
    value['scorer_number'] = [int(x.replace('scorer_', '')) - 0.5 for x in value['scorer']]
    value['x'] = value['x'] + (value['scorer_number'] / 5 - 0.5)

    xticks, xlabel = [], []
    for i, target_dataset in enumerate(target_datasets):

        dataset_name = target_dataset
        xticks += [i]
        xlabel += [target_dataset.replace('_', ' ').upper()]
        value = df[df['target'] == dataset_name]
        consensus_value = list(value[value['scorer'] == 'consensus']['f1_macro'])[0]


        scorers_all_values = value[value['scorer'] != 'consensus']['f1_macro']
        scorers_value = scorers_all_values.mean()
        min_scorer, max_scorer = scorers_value - scorers_all_values.min(), scorers_all_values.max() - scorers_value
        if i == 0:
            ax[k].bar(i - width / 2, consensus_value, width=width, edgecolor='black',
                      color="#ED5C79", label='Consensus')
            ax[k].bar(i + width / 2, scorers_value, width=width, edgecolor='black',
                      color="#fdb0ff", label='Average Scorer')
            ax[k].errorbar(i + width / 2, scorers_value, np.array([min_scorer, max_scorer]).reshape(
                -1, 1),
                           color="black", label='Best/worse scorer')

        else:
            ax[k].bar(i - width / 2, consensus_value, width=width, edgecolor='black',
                      color="#ED5C79")
            ax[k].bar(i + width / 2, scorers_value, width=width, edgecolor='black',
                      color="#fdb0ff")
            ax[k].errorbar(i + width / 2, scorers_value, np.array([min_scorer, max_scorer]).reshape(
                -1, 1),
                           color="black")

    ax[k].set(xlabel='', ylabel=f'{source_dataset.upper()} \n Macro F1', ylim=(0, 1))
    ax[k].spines['right'].set_visible(False)
    ax[k].spines['top'].set_visible(False)
    if k == 1:
        plt.xticks(xticks, xlabel)

plt.subplots_adjust(top=0.9)
handles, labels = ax[k].get_legend_handles_labels()
labels = [x.replace('_', ' ').capitalize() for x in labels]
legend = fig.legend(handles, labels, loc='upper center', ncol=3)
legend.get_frame().set_facecolor('none')
legend.get_frame().set_linewidth(0.0)
params = {'legend.fontsize': 18,
          'legend.markerscale': 2,
          'font.size': 20,
          'axes.titlesize': 20,
          }
plt.rcParams.update(params)
plt.show()
