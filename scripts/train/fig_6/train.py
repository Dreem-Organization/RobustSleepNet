"""
Code to finetune the models with an increasing number of records and to generate the FT learning curve from fig_6
"""

import hashlib
import json
import os
import shutil
import random as rd
import pandas as pd

from scripts.settings import (
    EXPERIMENTS_DIRECTORY,
    MASS_SETTINGS,
    MESA_SETTINGS,
    SHHS_SETTINGS,
    MROS_SETTINGS,
    DODO_SETTINGS,
    DODH_SETTINGS,
    CAP_SETTINGS,
    SLEEP_EDF_SETTINGS
)
from robust_sleep_net.utils.run_supervised_experiments import run_experiment


def memmap_hash(memmap_description):
    return hashlib.sha1(json.dumps(memmap_description).encode()).hexdigest()[:10]


TEMPORAL_CONTEXT = 21
EPOCHS = 100
PATIENCE = 10
memmaps_description = {}
splits = {}
EXPERIMENT_OUTPUT_FOLDER = EXPERIMENTS_DIRECTORY + "/fig_6/"
SPLITS_FOLDER = "scripts/train/config/supervised_learning_split/"
settings = {
    "mass": MASS_SETTINGS,
    "dodo": DODO_SETTINGS,
    "dodh": DODH_SETTINGS,
    "cap": CAP_SETTINGS,
    "mesa": MESA_SETTINGS,
    "mros": MROS_SETTINGS,
    "sleep_edf": SLEEP_EDF_SETTINGS,
    "shhs": SHHS_SETTINGS,
}
for dataset in settings:
    memmaps_description[dataset] = json.load(
        open(f"scripts/train/config/memmaps_description/{dataset}.json")
    )

for split in os.listdir(SPLITS_FOLDER):
    splits_name = split.replace(".json", "")
    splits[splits_name] = json.load(open(f"{SPLITS_FOLDER}/{split}"))

# model
model_description = json.load(open("scripts/train/config/model_settings/description.json"))
model_normalization = json.load(open("scripts/train/config/model_settings/normalization.json"))

force = False
num_workers = 4
DATASET_SPLIT_SEED = 2020
datasets_split = {}

trainer_parameters = {
    "type": "base",
    "args": {
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "num_workers": num_workers,
        "optimizer": {"type": "adam", "args": {"lr": 1e-4, "amsgrad": True}},
        "loss": {"type": "cross_entropy_with_weights", "args": {"inverse_frequency_weighting": True}}
    },
}

pretrained_model_folder = "pretrained_model"
pretrained_models_description = pd.read_csv(f"{pretrained_model_folder}/description.csv")
model_for_dataset = {}
for i, experiment in pretrained_models_description.iterrows():
    exp_id = experiment["experiment_id"]
    model_for_dataset[experiment["target_dataset"]] = f"{pretrained_model_folder}/{exp_id}/"

PRETRAINED_FOLDER = 'pretrained_model'
pretrained_models = [x for x in os.listdir(PRETRAINED_FOLDER)]
RECORDS_TO_USE = [2, 3, 5, 10, 20, 30, 40, 50, 75]

for split_name, split in splits.items():
    dataset = split["source_dataset"]
    for n_records in RECORDS_TO_USE:
        if dataset in model_for_dataset:
            checkpoint = {
                "directory": PRETRAINED_FOLDER,
                "net_to_load": model_for_dataset[dataset],
            }
            outfolder = run_experiment(
                settings=settings[dataset],
                memmaps_description=memmaps_description[dataset],
                temporal_context=TEMPORAL_CONTEXT,
                trainer=trainer_parameters,
                normalization=model_normalization,
                model=model_description,
                transform=None,
                split=split,
                SEED=2020,
                save_folder=f"{EXPERIMENT_OUTPUT_FOLDER}/{split_name}/{n_records}_record",
                checkpoint=checkpoint,
                max_training_records=n_records,
                error_tolerant=True
            )
            for folder in outfolder:
                shutil.rmtree(os.path.join(folder, 'training'))
                shutil.rmtree(os.path.join(folder, 'base_experiment'))
