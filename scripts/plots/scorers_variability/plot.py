import pandas as pd
import seaborn as sns

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
df = pd.read_csv('scripts/evaluation/plots/scorers_variability/results.csv')
df.groupby(['scorer','source']).std()['f1_macro']

target_datasets = df['target'].unique()
target_datasets_order = {k:i for i,k in enumerate(target_datasets)}
source_datasets = ['dodo','dodh']
target_datasets = [x for x in target_datasets if x not in source_datasets]



for k,source_dataset in enumerate(source_datasets):
    print(k)
    df = pd.read_csv('scripts/evaluation/plots/scorers_variability/results.csv')
    df = df[df['source'] == source_dataset]


    cmap = LinearSegmentedColormap.from_list("lol", ["#fff7fa", "#ED5C79"])
    value = df[df['scorer'] != 'consensus']
    print(value['scorer'].unique())
    value['x'] = [target_datasets_order[x] for x in value['target']]
    value['scorer_number'] = [int(x.replace('scorer_','')) - 0.5 for x in value['scorer']]
    value['x'] = value['x'] + (value['scorer_number']/5 - 0.5)
    swm_plot = sns.scatterplot("x","f1_macro",data = value,
                hue='scorer',hue_order=['scorer_1','scorer_2', 'scorer_3','scorer_4','scorer_5'],
                  palette=[cmap((x + 1)/5) for x in range(5)],edgecolor='black',linewidth=0.5,
                               ax=ax[k],
                               s=100,alpha = 0.9)
    swm_plot.legend_.remove()
    xticks,xlabel = [], []
    for i,target_dataset in enumerate(target_datasets):

        dataset_name = target_dataset
        xticks += [i]
        xlabel += [target_dataset.replace('_',' ').upper()]
        value = df[df['target'] == dataset_name]
        consensus_value = list(value[value['scorer'] == 'consensus']['f1_macro'])[0]
        scorers_value = value[value['scorer'] != 'consensus']['f1_macro'].mean()
        if i == 0:
            print('ok', k)
            ax[k].hlines(consensus_value,i - 0.5,i + 0.5,linestyles='dashed',
                    color="black",label='Consensus')
            ax[k].hlines(scorers_value,i - 0.5,i + 0.5,linestyles='dashed',
                    color="#ED5C79",label='Average Scorer')

        else:
            print('ok', k)
            ax[k].hlines(consensus_value,i - 0.5,i + 0.5,linestyles='dashed',
                    color="black")
            ax[k].hlines(scorers_value,i - 0.5,i + 0.5,linestyles='dashed',
                    color="#ED5C79")

    ax[k].set(xlabel='', ylabel=f'{source_dataset.upper()} \n Macro F1', ylim=(0, 1))
    if k == 1:
        plt.xticks(xticks, xlabel)

plt.subplots_adjust(top=0.9)
handles, labels = ax[k].get_legend_handles_labels()
labels =[x.replace('_', ' ').capitalize() for x in labels]
legend = fig.legend(handles[1:], labels[1:], loc='upper center',ncol = 7)
legend.get_frame().set_facecolor('none')
legend.get_frame().set_linewidth(0.0)
params = {'legend.fontsize': 18,
          'legend.markerscale': 2,
        'legend.markerscale': 2,
        'font.size' : 20,
          'axes.titlesize':20
          }
plt.rcParams.update(params)
plt.show()

