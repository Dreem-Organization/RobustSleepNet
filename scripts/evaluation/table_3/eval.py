"""
Compute the per epoch performance of a model trained with annotations from a single scorer on DODO or DODH and evaluated on the other datasets.
"""

import json
import os
import pandas as pd
from scripts.settings import EXPERIMENTS_DIRECTORY
from robust_sleep_net.utils import score_functions

results = []
base = EXPERIMENTS_DIRECTORY + "/scorers_transfer_eval/"
folders = [x for x in os.listdir(base) if os.path.isdir(f"{base}/{x}/")]
for experiment in folders:
    source, target = tuple(experiment.split("_to_"))

    for folder in os.listdir(f"{base}/{experiment}/"):
        hynograms = []
        predicted = []
        records = []
        experiment_folder = f"{base}/{experiment}/{folder}/"
        if "hypnograms.json" in os.listdir(experiment_folder):
            file = f"{experiment_folder}/hypnograms.json"
            hypno = json.load(open(file))
            for record, values in hypno.items():
                records += [record]
                hynograms += values["target"]
                predicted += values["predicted"]
            description = file = f"{experiment_folder}/description.json"
            description = json.load(open(description))
            dataset_name = description["dataset_parameters"]["split"]["test"][0].split("/")[5]
            training_set_size = len(description["dataset_parameters"]["split"]["train"]) + len(
                description["dataset_parameters"]["split"]["val"]
            )
            result_for_folder = {
                "source": source,
                "target": target,
                "scorer": description['dataset_parameters'].get("hypnogram_filename",
                                                                "consensus").replace('.mm', ''),
                "training_set_size": training_set_size,
                "records in dataset": len(records),
                "experiment_id": experiment,
            }
            for metric, f in score_functions.items():
                result_for_folder[metric] = f(hynograms, predicted)
            results += [result_for_folder]

df = pd.DataFrame(results)
df.sort_values("training_set_size")
df.to_csv(base + "results.csv")
