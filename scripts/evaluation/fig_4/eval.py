"""
Evaluation with channel ablation
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
EXPERIMENT_OUTPUT_FOLDER = EXPERIMENTS_DIRECTORY + "/fig_4/"
EXPERIMENT_OUTPUT_FOLDER = "/media/antoine/2Tb-ext/papier_transfer_learning/channels_ablation/"
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

channels_ablation = {}
channels_ablation['dodo'] = [
    {
        "eeg": [0, 1],
        "name": 'Cx_Mx'
    },
    {
        "eeg": [4],
        "name": 'F4_O2'
    },
    {
        "eeg": [4],
        "name": 'Single Channel'
    },
    {
        "eeg": [5],
        "name": 'F3_O1'
    },
    {
        "eeg": [0, 1, 2, 3, 4, 5, 6, 7],
        "name": 'EEG'
    },
    {
        "eeg": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
        "name": 'EEG+EOG'
    },
    {
        "eeg": [8],
        "name": 'EMG'
    },
    {
        "eeg": [9, 10],
        "name": 'EOG'
    },

]

channels_ablation['dodh'] = [
    {
        "eeg": [0],
        "name": 'C3_M2'
    },
    {
        "eeg": [4],
        "name": 'F4_O2'
    },
    {
        "eeg": [4],
        "name": 'Single Channel'
    },
    {
        "eeg": [5],
        "name": 'F3_O1'
    },
    {
        "eeg": [6],
        "name": 'FP1_F3'
    },

    {
        "eeg": [2],
        "name": 'F3_F4'
    },
    {
        "eeg": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "name": 'EEG'
    },
    {
        "eeg": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14],
        "name": 'EEG+EOG'
    },
    {
        "eeg": [12],
        "name": 'EMG'
    },
    {
        "eeg": [13, 14],
        "name": 'EOG'
    },

]

channels_ablation['mass'] = [
    {
        "eeg": [0],
        "name": 'C4_O1'
    },
    {
        "eeg": [0],
        "name": 'Single Channel'
    },
    {
        "eeg": [1],
        "name": 'F4_EOG'
    },

    {
        "eeg": [2],
        "name": 'F8_Cz'
    },
    {
        "eeg": [3],
        "name": 'EOG'
    },

    {
        "eeg": [4],
        "name": 'EMG'
    },
    {
        "eeg": [0,2],
        "name": 'EEG'
    },
    {
        "eeg": [0, 1, 2, 3],
        "name": 'EEG+EOG'
    }

]

channels_ablation['sleep_edf'] = [
    {
        "eeg": [0,1],
        "name": 'EEG'
    },
    {
        "eeg": [0],
        "name": 'Fpz-Cz'
    },
    {
        "eeg": [0],
        "name": 'Single Channel'
    },
    {
        "eeg": [1],
        "name": 'Pz-Oz'
    },
    {
        "eeg": [2],
        "name": 'EOG'
    },
    {
        "eeg": [0,1,2],
        "name": 'EEG+EOG'
    }

]

channels_ablation['cap'] = [
    {
        "eeg": [0],
        "name": 'C4-A1'
    },
    {
        "eeg": [0],
        "name": 'Single Channel'
    },
    {
        "eeg": [1],
        "name": 'F3-C4'
    },
    {
        "eeg": [0,1],
        "name": 'EEG'
    },
    {
        "eeg": [2],
        "name": 'EOG'
    }

]

channels_ablation['mesa'] = [
    {
        "eeg": [0,1],
        "name": 'EOG'
    },
    {
        "eeg": [2,3,4],
        "name": 'EEG'
    },
    {
        "eeg": [2],
        "name": 'Single Channel'
    },
    {
        "eeg": [2],
        "name": 'EEG1'
    },
    {
        "eeg": [3],
        "name": 'EEG2'
    },
    {
        "eeg": [4],
        "name": 'EEG3'
    },
    {
        "eeg": [0,1,2,3,4],
        "name": 'EEG+EOG'
    },
    {
        "eeg": [5],
        "name": 'EMG'
    }

]


channels_ablation['mros'] = [
    {
        "eeg": [0],
        "name": 'C4-M1'
    },
    {
        "eeg": [0],
        "name": 'Single Channel'
    },
    {
        "eeg": [1],
        "name": 'C3-M2'
    },
    {
        "eeg": [0,1,2,3],
        "name": 'EEG'
    },
    {
        "eeg": [4],
        "name": 'EOG'
    },
    {
        "eeg": [0,1,2,3,4],
        "name": 'EEG+EOG'
    },
    {
        "eeg": [5,6],
        "name": 'EMG'
    }

]

channels_ablation['shhs'] = [
    {
        "eeg": [0],
        "name": 'EEG1'
    },
    {
        "eeg": [0],
        "name": 'Single Channel'
    },
    {
        "eeg": [1],
        "name": 'EEG2'
    },
    {
        "eeg": [2],
        "name": 'EMG'
    },
    {
        "eeg": [0,1],
        "name": 'EEG'
    }

]


channels_ablation['shhs'] = [
    {
        "eeg": [3,4],
        "name": 'EOG'
    }

]

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

for split_name, split in splits.items():
    dataset = split["source_dataset"]
    if dataset in channels_ablation:
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
            validate_with_ablation_modalities=channels_ablation[dataset],
            error_tolerant=True
        )

results = []
for dataset in os.listdir(EXPERIMENT_OUTPUT_FOLDER):
    if '.' not in dataset:

        sub_experiments = os.listdir(f"{EXPERIMENT_OUTPUT_FOLDER}/{dataset}/")
        hypno = {'baseline':{}}

        for sub_experiment in sub_experiments:
            ablations_folder = f"{EXPERIMENT_OUTPUT_FOLDER}/{dataset}/" \
                               f"{sub_experiment}/hypnograms_ablation/"
            file = f"{EXPERIMENT_OUTPUT_FOLDER}/{dataset}/" \
                               f"{sub_experiment}/hypnograms.json"
            hypno['baseline'].update(json.load(open(file)))
            for ablation in os.listdir(ablations_folder):
                try:
                    file = f"{ablations_folder}/{ablation}"
                    if ablation not in hypno:
                        hypno[ablation] = {}
                    hypno[ablation].update(json.load(open(file)))

                    used_records = []
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
