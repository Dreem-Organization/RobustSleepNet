"""
Add the individual scorers hypnogram to and already generated memmaps file
"""

import json
import os

import numpy as np


def load_hypnograms():
    return {
        "dodh": json.load(open("scripts/train/scorers_transferability/scoring/dodh.json")),
        "dodo": json.load(open("scripts/train/scorers_transferability/scoring/dodo.json"))
    }


def add_scorers_hypnograms(memmap_folder, dataset, scorer):
    replacement_hypnograms = load_hypnograms()[dataset][scorer]
    print(dataset, scorer)
    print(list(replacement_hypnograms.keys()))
    for folder in os.listdir(memmap_folder):
        if '.' not in folder:
            record_folder = f"{memmap_folder}/{folder}/"
            properties = json.load(open(f'{record_folder}/properties.json'))['eeg']
            padding = properties['padding'] // 30

            hypnogram_for_scorer = [-1] * padding + replacement_hypnograms[folder] + [-1] * padding
            hypnogram_for_scorer = np.array(hypnogram_for_scorer)
            hypnogram_memmap = np.memmap(
                f"{record_folder}/{scorer}.mm",
                dtype="float32",
                mode="w+",
                shape=hypnogram_for_scorer.shape,
            )
            hypnogram_memmap[:] = hypnogram_for_scorer
