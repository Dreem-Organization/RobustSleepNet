"""
LfS row of the table 2
"""

import json

from robust_sleep_net.utils import score_functions
import os
import pandas as pd
import numpy as np
from scripts.settings import EXPERIMENTS_DIRECTORY
experiments_folder = EXPERIMENTS_DIRECTORY + "/supervised_learning/"
results = []
for experiment in os.listdir(experiments_folder):

    result_folder = os.path.join(experiments_folder, experiment)
    if os.path.isdir(result_folder):
        list_of_results = os.listdir(result_folder)
        list_of_results.sort()
        hynograms = []
        predicted = []
        blacklist = []
        added_records = []
        hypno = {}
        for dataset in os.listdir(result_folder):
            sub_experiments = os.listdir(f"{result_folder}/{dataset}/")
            file = f"{result_folder}/{dataset}/hypnograms.json"
            x = json.load(open(file))

            for record, hypnogram in x.items():
                if record not in added_records:
                    hypno[record] = hypnogram
                    added_records += [record]
                    hypno_array = np.array(hypnogram["target"][30:-30])
                    hynograms += hypnogram["target"][30:-30]
                    predicted += hypnogram["predicted"][30:-30]


        x_tp = {"dataset": experiment, "records": len(added_records)}
        for metric, f in score_functions.items():
            x_tp.update({metric: f(hynograms, predicted)})
        results += [x_tp]

df = pd.DataFrame(results)
df.to_csv(f"{experiments_folder}/results.csv")
