import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

params = {'legend.fontsize': 20,
          'legend.markerscale': 1,
          'font.size': 20,
          'axes.titlesize': 20
          }
plt.rcParams.update(params)
fig, ax = plt.subplots(1, 1)
df = pd.read_csv('scripts/plots/fig_4/results.csv')

df['channels'] = [x.replace('ablation_', '') for x in df['channels']]
datasets = ['dodo', 'dodh', 'mass', 'sleep edf', 'mros', 'mesa', 'shhs', 'cap']
datasets = {k: v for v, k in enumerate(datasets)}
df = df[
    [x in ['Single Channel', "baseline", "EEG", "EOG", "EEG-EOG"] for x in df['channels']]]
df = df[
    [x in datasets for x in df['dataset']]]

cmap = LinearSegmentedColormap.from_list("lol", ["#fff7fa", "#ED5C79"])
from matplotlib import cm
import numpy as np
import seaborn as sns
cmap = cm.get_cmap('Set3', 5)
cmap = sns.color_palette(n_colors=5)
cmap = [elt + (0.5,) for elt in cmap]
# cmap = ["#d7dbdd99", "#dc763399", '#e5986699', '#f8c47199', '#85c1e999']
# cmap = [
#         (0.8666666666666667, 0.5176470588235295, 0.3215686274509804, 0.5),
#         (0.3333333333333333, 0.6588235294117647, 0.40784313725490196, 0.5),
#         (0.7686274509803922, 0.3058823529411765, 0.3215686274509804, 0.5),
#         (0.5058823529411764, 0.4470588235294118, 0.7019607843137254, 0.5),
#         (0.5764705882352941, 0.47058823529411764, 0.3764705882352941, 0.5)]
markers = [ 'EOG', "Single Channel", "EEG", 'EEG-EOG', "baseline"]
colors = {v: k for k, v in enumerate(markers)}
xticks, xlabel = [], []

params = {'legend.fontsize': 24,
          'legend.markerscale': 1,
          'font.size': 20,
          'axes.titlesize': 20,
          'font.family': 'Open Sans'

          }
plt.rcParams.update(params)
for i, target_dataset in enumerate(datasets):
    width = 0.15
    x_dataset = datasets[target_dataset]

    df_for_dataset = df[df['dataset'] == target_dataset]
    signals_for_datasets = [channel for channel in markers if len(df_for_dataset[df_for_dataset['channels'] == channel])
                            == 1]

    x_start_dataset = x_dataset
    x_end_dataset = x_dataset + (len(signals_for_datasets) - 1) * width

    xticks.append((x_start_dataset + x_end_dataset) / 2)
    xlabel.append(target_dataset.upper())
    for k, signal in enumerate(signals_for_datasets):
        if signal in signals_for_datasets:
            x = x_start_dataset + k * width
            value = df[df['dataset'] == target_dataset]
            consensus_value = list(value[value['channels'] == signal]['f1_macro'])[0]
            color = cmap[colors[signal]]
            sct_plot = plt.bar(x, consensus_value, color=color, edgecolor='black', label=signal,
                               width=width)

plt.xticks(xticks, xlabel)
ax.set(xlabel='Macro F1', ylabel="", ylim=(0, 1))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_facecolor('white')
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
import numpy as np

handles, labels = ax.get_legend_handles_labels()

labels, handles_idx = np.unique(labels, return_index=True)
labels_to_handle = {k: v for k, v in zip(labels, handles_idx)}
handles = [handles[labels_to_handle[i]] for i in markers]
legend = fig.legend(handles, markers, loc='upper center', ncol=6)

frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('white')

plt.xlabel('')
plt.ylabel('Macro F1')
plt.show()
