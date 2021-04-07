import json
import os
from scripts.settings import  EXPERIMENTS_DIRECTORY
import pandas as pd
import shutil
from robust_sleep_net.utils import score_functions

results = []
BASE_FOLDER = EXPERIMENTS_DIRECTORY + "/cross_dataset_eval/"
SUPERVISED_LEARNING_FOLDER = EXPERIMENTS_DIRECTORY + "/supervised_learning/"
shutil.move(os.path.join(SUPERVISED_LEARNING_FOLDER,"results.csv"),'LfS_results.csv')

folders = [x for x in os.listdir(BASE_FOLDER) if os.path.isdir(f"{BASE_FOLDER}/{x}/")]
for experiment in folders:
    source, target = tuple(experiment.split("_to_"))
    hynograms = []
    predicted = []
    records = []
    for folder in os.listdir(f"{BASE_FOLDER}/{experiment}/"):
        experiment_folder = f"{BASE_FOLDER}/{experiment}/{folder}/"
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
            "training_set_size": training_set_size,
            "records in " "dataset": len(records),
            "experiment_id": experiment,
        }
        for metric, f in score_functions.items():
            result_for_folder[metric] = f(hynograms, predicted)
        results += [result_for_folder]

df = pd.DataFrame(results)
df.sort_values("training_set_size")
df.to_csv("transfer_between_datasets.csv")
