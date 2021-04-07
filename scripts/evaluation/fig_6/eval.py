"""
Direct transfer learning curve
"""

import json
from robust_sleep_net.utils import score_functions
import os
import pandas as pd
from scripts.settings import EXPERIMENTS_DIRECTORY
results = []
base = EXPERIMENTS_DIRECTORY + "/fig_6/"

folders = [x for x in os.listdir(base) if os.path.isdir(f"{base}/{x}/")]
for experiment in folders:
    hynograms = []
    predicted = []
    records = []
    if "hypnograms.json" in os.listdir(f"{base}/{experiment}/"):
        file = f"{base}/{experiment}/hypnograms.json"
        hypno = json.load(open(file))
        for record, values in hypno.items():
            records += [record]
            hynograms += values["target"]
            predicted += values["predicted"]
        description = file = f"{base}/{experiment}/description.json"
        description = json.load(open(description))
        dataset_name = description["dataset_parameters"]["split"]["test"][0].split("/")[5]
        training_set_size = len(description["dataset_parameters"]["split"]["train"]) + len(
            description["dataset_parameters"]["split"]["val"]
        )
        result_for_folder = {
            "dataset": dataset_name,
            "training_set_size": training_set_size,
            "records in " "dataset": len(records),
            "experiment_id": experiment,
        }
        for metric, f in score_functions.items():
            result_for_folder[metric] = f(hynograms, predicted)
        results += [result_for_folder]

df = pd.DataFrame(results)
df.sort_values("training_set_size")
df.to_csv(base + "results.csv")
