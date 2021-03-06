import hashlib
import json
from scripts.settings import (
    EXPERIMENTS_DIRECTORY,
    MASS_SETTINGS,
    MESA_SETTINGS,
    SHHS_SETTINGS,
    MROS_SETTINGS,
    DODO_SETTINGS,
    DODH_SETTINGS,
    SLEEP_EDF_IN_BED_SETTINGS,
    CAP_SETTINGS,
)
import os

from robust_sleep_net.utils.run_supervised_experiments import run_experiment


def memmap_hash(memmap_description):
    return hashlib.sha1(json.dumps(memmap_description).encode()).hexdigest()[:10]


TEMPORAL_CONTEXT = 21
EPOCHS = 100
PATIENCE = 5
memmaps_description = {}
splits = {}
EXPERIMENT_OUTPUT_FOLDER = EXPERIMENTS_DIRECTORY + "/table_2_LfS/"
SPLITS_FOLDER = "scripts/train/config/supervised_learning_split/"
settings = {
    "mass": MASS_SETTINGS,
    "dodo": DODO_SETTINGS,
    "dodh": DODH_SETTINGS,
    "sleep_edf_in_bed": SLEEP_EDF_IN_BED_SETTINGS,
    "sleep_edf": SLEEP_EDF_IN_BED_SETTINGS,
    "cap": CAP_SETTINGS,
    "mesa": MESA_SETTINGS,
    "mros": MROS_SETTINGS,
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
        "optimizer": {"type": "adam", "args": {"lr": 1e-3, "amsgrad": True}},
        "loss": {"type": "cross_entropy_with_weights", "args": {}},
    },
}

for split_name, split in splits.items():
    dataset = split["source_dataset"]
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
        )
