"""
Evaluate the pretrained model on DODO and DODH by using a specific scorer instead of the consensus as the ground truth
"""

import hashlib
import json
import os

import pandas as pd

from scripts.settings import (
    EXPERIMENTS_DIRECTORY,
    MASS_SETTINGS,
    MESA_SETTINGS,
    SHHS_SETTINGS,
    MROS_SETTINGS,
    DODO_SETTINGS,
    DODH_SETTINGS,
    SLEEP_EDF_SETTINGS,
    CAP_SETTINGS,
)
from robust_sleep_net.utils import score_functions
from robust_sleep_net.utils.run_supervised_experiments import run_experiment


def memmap_hash(memmap_description):
    return hashlib.sha1(json.dumps(memmap_description).encode()).hexdigest()[:10]


TEMPORAL_CONTEXT = 21
EPOCHS = 100
PATIENCE = 5
memmaps_description = {}
splits = {}
EXPERIMENT_OUTPUT_FOLDER = EXPERIMENTS_DIRECTORY + "/supervised_learning_per_scorer/"
SPLITS_FOLDER = "scripts/train/config/supervised_learning_split/"
settings = {
    "mass": MASS_SETTINGS,
    "dodo": DODO_SETTINGS,
    "dodh": DODH_SETTINGS,
    "sleep_edf": SLEEP_EDF_SETTINGS,
    "cap": CAP_SETTINGS,
    "mesa": MESA_SETTINGS,
    "mros": MROS_SETTINGS,
    "shhs": SHHS_SETTINGS,
}

for dataset in settings:
    memmaps_description[dataset] = json.load(
        open(f"scripts/train/memmaps_description/{dataset}.json")
    )

for split in os.listdir(SPLITS_FOLDER):
    splits_name = split.replace(".json", "")
    splits[splits_name] = json.load(open(f"{SPLITS_FOLDER}/{split}"))

# model
model_description = json.load(open("scripts/train/model_settings/description.json"))
model_normalization = json.load(open("scripts/train/model_settings/normalization.json"))

force = False
num_workers = 4
DATASET_SPLIT_SEED = 2020
datasets_split = {}

trainer_parameters = {
    "type": "base",
    "args": {
        "epochs": 0,
        "patience": PATIENCE,
        "num_workers": num_workers,
        "optimizer": {"type": "adam", "args": {}},
        "loss": {"type": "cross_entropy_with_weights", "args": {}}
    },
}

pretrained_model_folder = "pretrained_model"
pretrained_models_description = pd.read_csv(f"{pretrained_model_folder}/description.csv")
model_for_dataset = {}
for i, experiment in pretrained_models_description.iterrows():
    exp_id = experiment["experiment_id"]
    model_for_dataset[experiment["target_dataset"]] = f"{pretrained_model_folder}/{exp_id}/"
scorers = ['scorer_1.mm','scorer_2.mm','scorer_3.mm','scorer_4.mm','scorer_5.mm']
for split_name, split in splits.items():
    dataset = split["source_dataset"]
    if dataset in ['dodo','dodh']:
        for scorer in scorers:
            checkpoint = {
                "directory": model_for_dataset[dataset],
                "net_to_load": "best_model.gz",
            }
            run_experiment(
                settings=settings[dataset],
                memmaps_description=memmaps_description[dataset],
                temporal_context=TEMPORAL_CONTEXT,
                trainer=trainer_parameters,
                normalization=model_normalization,
                model=model_description,
                transform=None,
                split=split,
                save_folder=f"{EXPERIMENT_OUTPUT_FOLDER}/{split_name}/",
                checkpoint=checkpoint,
                error_tolerant=True,
                hypnogram_filename_test = scorer
            )

results = []
for dataset in ['dodo','dodh']:
    if '.' not in dataset:

        sub_experiments = os.listdir(f"{EXPERIMENT_OUTPUT_FOLDER}/{dataset}/")
        hypno = {scorer.replace('.mm',''):{} for scorer in scorers}

        for sub_experiment in sub_experiments:
            try:
                file = f"{EXPERIMENT_OUTPUT_FOLDER}/{dataset}/" \
                                   f"{sub_experiment}/hypnograms.json"
                scorer = json.load(open(f"{EXPERIMENT_OUTPUT_FOLDER}/{dataset}/" \
                                   f"{sub_experiment}/description.json"))
                scorer = scorer['dataset_parameters'].get('hypnogram_filename_test',
                                                          'consensus').replace('.mm','')
                hypno[scorer].update(json.load(open(file)))
            except FileNotFoundError:
                pass


        import numpy as np

        for ablation, hypnogram_for_ablations in hypno.items():
            hynograms = []
            predicted = []
            blacklist = []
            records = []
            used_records = []
            for record, values in hypnogram_for_ablations.items():
                used_records += [record]
                hypno_array = np.array(values["target"][30:-30])
                hynograms += values["target"][30:-30]
                predicted += values["predicted"][30:-30]

            x_tp = {"dataset": dataset, "records": len(used_records), "channels": ablation.replace(
                '.json', '')}
            for metric, f in score_functions.items():
                x_tp.update({metric: f(hynograms, predicted)})
            results += [x_tp]

import pandas as pd

df = pd.DataFrame(results)
df.to_csv(f"{EXPERIMENT_OUTPUT_FOLDER}/results.csv")
