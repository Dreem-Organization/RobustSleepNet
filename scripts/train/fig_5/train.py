"""
Train models for generalization and easiness of each datasets
"""

import hashlib
import json

import pandas as pd

from robust_sleep_net.logger.logger import log_experiment
from robust_sleep_net.preprocessings.h5_to_memmap import h5_to_memmaps
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
from robust_sleep_net.utils.train_test_val_split import train_test_val_split


def memmap_hash(memmap_description):
    return hashlib.sha1(json.dumps(memmap_description).encode()).hexdigest()[:10]


TEMPORAL_CONTEXT = 21
EPOCHS = 100
PATIENCE = 5
memmaps_description = {}
EXPERIMENT_OUTPUT_FOLDER = EXPERIMENTS_DIRECTORY + "/cross_dataset_train/"
CROSS_EXPERIMENT_OUTPUT_FOLDER = EXPERIMENTS_DIRECTORY + "/cross_dataset_eval/"
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
        open(f"scripts/train/config/memmaps_description/{dataset}.json")
    )

# model
model_description = json.load(open("scripts/train/config/model_settings/description.json"))
model_normalization = json.load(open("scripts/train/config/model_settings/normalization.json"))

import os

force = False
num_workers = 4
DATASET_SPLIT_SEED = 2020
datasets_split = {}
datasets = ["sleep_edf", "mass"]
max_records_per_dataset = 250

trainer_parameters = {
    "type": "base",
    "args": {
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "num_workers": num_workers,
        "optimizer": {"type": "adam", "args": {"lr": 1e-3, "amsgrad": True}},
        "loss": {"type": "cross_entropy_with_weights", "args": {"weight": [0.5, 2, 1, 1, 1]},},
    },
}
for i in range(len(datasets)):
    source_dataset = datasets[i]
    target_datasets = [dataset for dataset in datasets if dataset != source_dataset]

    for dataset_name in datasets:
        setting = settings[dataset_name]
        records_in_dataset = json.load(
            open(f"scripts/train/config/perf_by_records/" f"{dataset_name}.json")
        )
        df = pd.DataFrame.from_dict(records_in_dataset, orient="index")
        df.columns = ["score"]
        df = df.sort_values("score", ascending=False)

        output_folders = {
            record.replace(".h5", ""): setting["h5_directory"] + record
            for record in os.listdir(setting["h5_directory"])
        }
        dataset_memmap_hash = memmap_hash(memmaps_description[dataset_name])
        if not force:
            if not os.path.exists(setting["memmap_directory"]):
                os.makedirs(setting["memmap_directory"])
            try:
                already_created_record = os.listdir(
                    setting["memmap_directory"] + dataset_memmap_hash + "/"
                )
                records_in_dataset = list(output_folders.values())
            except FileNotFoundError:
                pass

        records_in_dataset.sort()
        if dataset_name in datasets:
            memmaps_directory, groups_description, features_description = h5_to_memmaps(
                records=records_in_dataset,
                memmap_description=memmaps_description[dataset_name],
                memmap_directory=setting["memmap_directory"],
                num_workers=num_workers,
                error_tolerant=False,
            )
        else:
            memmaps_directory = setting["memmap_directory"] + "/" + dataset_memmap_hash + "/"

        if dataset_name == source_dataset:
            records_for_train_and_validation = list(df.index[:max_records_per_dataset])
            records_for_train_and_validation = [
                x for x in records_for_train_and_validation if x in os.listdir(memmaps_directory)
            ]
            records_for_train_and_validation = [
                memmaps_directory + record + "/"
                for record in records_for_train_and_validation
                if "." not in record
            ]
            datasets_split[dataset_name] = train_test_val_split(
                records_for_train_and_validation, 0.7, 0.0, 0.3, seed=DATASET_SPLIT_SEED
            )
        else:
            datasets_split[dataset_name] = train_test_val_split(
                [
                    memmaps_directory + record + "/"
                    for record in os.listdir(memmaps_directory)
                    if "." not in record
                ],
                0,
                1,
                0,
                seed=DATASET_SPLIT_SEED,
            )
        datasets_split[dataset_name] = {
            "train": datasets_split[dataset_name][0],
            "test": datasets_split[dataset_name][1],
            "validation": datasets_split[dataset_name][2],
        }

    results = {source_dataset: {}}
    train_split, validation_split, test_split = [], [], []
    for dataset in datasets:
        train_split += datasets_split[dataset]["train"]
        validation_split += datasets_split[dataset]["validation"]
        test_split += datasets_split[dataset]["test"]

    experiment_to_log = {
        "memmap_description": memmaps_description[source_dataset],
        "dataset_settings": settings[source_dataset],
        "trainer_parameters": trainer_parameters,
        "normalization_parameters": model_normalization,
        "net_parameters": model_description,
        "dataset_parameters": {
            "temporal_context": TEMPORAL_CONTEXT,
            "transform_parameters": [{"name": "eeg", "processing": []}],
            "split": {
                "train": train_split,
                "val": validation_split,
                "test": None,
                "parameters_init": train_split,
            },
        },
        "save_folder": EXPERIMENT_OUTPUT_FOLDER,
    }

    folder = log_experiment(**experiment_to_log, num_workers=num_workers, generate_memmaps=False)
    checkpoint = {"directory": folder, "net_to_load": "training/best_net"}
    trainer_eval = {
        "type": "base",
        "args": {
            "epochs": 0,
            "patience": 0,
            "num_workers": 0,
            "optimizer": {"type": "adam", "args": {"lr": 1e-3, "amsgrad": True}},
        },
    }

    for dataset in target_datasets:
        if dataset != source_dataset:
            result_folder = f"{CROSS_EXPERIMENT_OUTPUT_FOLDER}/{source_dataset}_to_{dataset}/"
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            experiment_to_log = {
                "memmap_description": memmaps_description[source_dataset],
                "dataset_settings": settings[source_dataset],
                "trainer_parameters": trainer_parameters,
                "normalization_parameters": model_normalization,
                "net_parameters": model_description,
                "dataset_parameters": {
                    "temporal_context": TEMPORAL_CONTEXT,
                    "transform_parameters": [{"name": "eeg", "processing": []}],
                    "split": {
                        "train": train_split,
                        "val": validation_split,
                        "test": datasets_split[dataset]["test"],
                        "parameters_init": train_split,
                    },
                },
                "save_folder": result_folder,
            }
            log_experiment(
                **experiment_to_log,
                num_workers=num_workers,
                generate_memmaps=False,
                checkpoint=checkpoint,
            )
