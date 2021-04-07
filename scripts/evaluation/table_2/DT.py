"""
Direct transfer row of the table 2
"""

from robust_sleep_net.utils import score_functions
import pandas as pd
import os
import json
import numpy as np
from scripts.settings import EXPERIMENTS_DIRECTORY
folder = EXPERIMENTS_DIRECTORY + "/DT/"

path_to_split = 'scripts/train/config/supervised_learning_split/'
results = []
experiments_list = []
for experiment in os.listdir(folder):
    if os.path.isdir(os.path.join(folder,experiment)):
        description = json.load(open(os.path.join(folder,experiment,"description.json")))
        dataset_name = description["dataset_parameters"]["split"]["test"][0].split("/")[5]
        experiments_list += [{"experiment_id":experiment,'dataset':dataset_name}]
experiments_list = pd.DataFrame(experiments_list)

for split in os.listdir(path_to_split):
    target = {}
    predicted = {}
    blacklist = []
    records = []
    split_description = json.load(open(os.path.join(path_to_split,split)))
    records_for_split = []
    if 'subjects' in split_description['args']:

        subjects = split_description['args']['subjects']
        for d in subjects:
            records_for_split += d["records"]

    dataset = split_description['source_dataset']

    df = experiments_list[experiments_list['dataset'] == dataset]

    if len(df) > 0:
        for experiment in df['experiment_id'].tolist():
            try:
                experiment_folder = os.path.join(folder,experiment)

                file = f"{experiment_folder}/hypnograms.json"
                hypno = json.load(open(file))
                for record,values in hypno.items():
                    if record not in target:
                        target[record] = hypno[record]['target']
                        predicted[record] = [hypno[record]['predicted']]
                    else:
                        predicted[record] += [hypno[record]['predicted']]
                records = list(hypno.keys())
                records.sort()

                used_records = []
            except FileNotFoundError:
                pass


        hypnograms = []
        predicted_hypnograms = []
        for record, values in target.items():
            if record in records_for_split or records_for_split == []:
                hypno_array = np.array(target[record][30:-30])
                first_sleep = np.where(hypno_array > 0)[0][0]
                last_sleep = np.where(hypno_array > 0)[0][-1]
                cut_off = len(hypno_array)
                if dataset == 'sleep_edf':
                    if len(hypno_array) - last_sleep >= 60:
                        cut_off = last_sleep + 60

                for i,values in enumerate(predicted[record]):
                    hypnograms += target[record][30:-30][:cut_off]
                    predicted_hypnograms += values[30:-30][:cut_off]

        results_tp = {'split':split.replace('.json',''),'dataset':dataset}
        for metric, f in score_functions.items():
            results_tp.update({metric:f(hypnograms, predicted_hypnograms)})

        results.append(results_tp)

results = pd.DataFrame(results)
results.to_csv(os.path.join(folder,'results.csv'))

