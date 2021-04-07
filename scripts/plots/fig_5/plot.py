import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

import copy
default_params = copy.deepcopy(plt.rcParams)
supervised = pd.read_csv("scripts/plots/fig_5/LfS_results.csv")
transfer = pd.read_csv("scripts/plots/fig_5/transfer_between_datasets.csv")
datasets = np.unique(transfer[["target"]])
datasets.sort()

datasets_list = ['DODO','DODH','MASS','SLEEP EDF',"MROS","MESA","SHHS","CAP"]
datasets_list_lwr = ['dodo','dodh','mass','sleep_edf','mros','mesa','shhs','cap']
results = np.zeros((len(datasets) + 2, len(datasets) + 2))
for i, source_dataset in enumerate(datasets_list_lwr):
    for j, target_dataset in enumerate(datasets_list_lwr):
        print(source_dataset, target_dataset)
        try:
            if i == j:
                results[i, j] = supervised[supervised["dataset"] == source_dataset][
                    "f1_macro"
                ].values[0]
            else:
                results[i, j] = transfer[
                    (transfer["source"] == source_dataset) * (transfer["target"] == target_dataset)
                    ]["f1_macro"].values[0]
        except:
            pass

norm_results = np.copy(results) * 100
for j, target_dataset in enumerate(datasets_list):
    pass
    #norm_results[:, j] = norm_results[:, j] / norm_results[j, j] * 100
    #norm_results = norm_results * 100

generalization = (np.sum(norm_results, 1) - 100) / (len(datasets) - 1)
easiness = (np.sum(norm_results, 0) - 100) / (len(datasets) - 1)
datasets = np.array([x.upper().replace("_", " ") for x in datasets])
norm_results[:, -1] = generalization
norm_results[-1, :] = easiness
mask = np.zeros((len(datasets) + 2, len(datasets) + 2))
mask[:, 8] = 1
mask[8] = 1
mask[9:, 9:] = 1
results = pd.DataFrame(norm_results)
results.index = datasets_list + ['', 'Easiness']
results.columns = datasets_list + ['', 'Generalization']

cmap = LinearSegmentedColormap.from_list("lol", ["#fff7fa", "#ED5C79"])

with sns.axes_style("white"):
    params = {
        'font.size':20,
        'axes.titlesize': 30,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.labelsize': 20,
        'font.family': 'Open Sans'

    }
    sns.set(rc=params)
    Z = np.array(results)
    ax = sns.heatmap(results, annot=True, fmt=".1f", cmap=cmap, cbar=False, linewidths=.5, mask=mask)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    plt.xlabel("Target datasets", labelpad=20)
    plt.ylabel("Source datasets")
    x_ticks = np.arange(0, 10) + 0.5
    x_ticks = x_ticks.tolist()
    #x_ticks = x_ticks[:-2] + x_ticks[-1:]
    ax.set_xticks(x_ticks)
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.set_facecolor('white')
    plt.show()
