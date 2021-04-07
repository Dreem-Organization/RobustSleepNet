from scripts.settings import DODH_SETTINGS, DODO_SETTINGS, SLEEP_EDF_SETTINGS, CAP_SETTINGS
from scripts.minimal_exemple.download_data import download_dodo, download_cap, download_dodh, download_sleep_edf

if __name__ == "__main__":
    # Download the four datasets
    download_dodo(DODO_SETTINGS)
    download_dodh(DODH_SETTINGS)

    download_sleep_edf(SLEEP_EDF_SETTINGS)
    download_cap(CAP_SETTINGS)
    import hashlib
    import json

    import pandas as pd

    from robust_sleep_net.logger.logger import log_experiment
    from robust_sleep_net.preprocessings.h5_to_memmap import h5_to_memmaps
    from scripts.settings import (
        EXPERIMENTS_DIRECTORY,
        DODO_SETTINGS,
        DODH_SETTINGS,
        SLEEP_EDF_SETTINGS,
        CAP_SETTINGS,
    )
    from robust_sleep_net.utils.train_test_val_split import train_test_val_split

    def memmap_hash(memmap_description):
        return hashlib.sha1(json.dumps(memmap_description).encode()).hexdigest()[:10]

    TEMPORAL_CONTEXT = 11
    EPOCHS = 10
    PATIENCE = 5
    memmaps_description = {}
    EXPERIMENT_OUTPUT_FOLDER = EXPERIMENTS_DIRECTORY + "/minimal_example/"
    settings = {
        "dodo": DODO_SETTINGS,
        "dodh": DODH_SETTINGS,
        "sleep_edf": SLEEP_EDF_SETTINGS,
        "cap": CAP_SETTINGS
    }
    for dataset in settings:
        memmaps_description[dataset] = json.load(
            open(f"scripts/minimal_exemple/config/memmaps_description/{dataset}.json")
        )

    # model
    model_description = json.load(
        open("scripts/minimal_exemple/config/model_settings/description.json"))
    model_normalization = json.load(
        open("scripts/minimal_exemple/config/model_settings/normalization.json"))

    import os

    force = False
    num_workers = 5
    DATASET_SPLIT_SEED = 2020
    datasets_split = {}
    datasets = ["sleep_edf", "dodh", 'dodo', 'cap']
    n_records_per_dataset = 50

    trainer_parameters = {
        "type": "base",
        "args": {
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "num_workers": num_workers,
            "optimizer": {"type": "adam", "args": {"lr": 0.001, "amsgrad": True}},
            "loss": {"type": "cross_entropy_with_weights", "args": {"inverse_frequency_weighting": True}},
        },
    }

    for i in range(len(datasets)):
        target_dataset = datasets[i]
        source_datasets = [dataset for dataset in datasets if dataset != target_dataset]
        record_per_dataset = {}
        records_to_sample = (len(datasets) - 1) * n_records_per_dataset
        sampled_records = 0
        record_to_sample_from_dataset = {}
        for dataset_name in datasets:
            record_per_dataset[dataset_name] = len(json.load(
                open(f"scripts/train/config/perf_by_records/{dataset_name}.json")
            ))
            if dataset_name != target_dataset:
                record_to_sample_from_dataset[dataset_name] = min(record_per_dataset[dataset_name],
                                                                  n_records_per_dataset)
                sampled_records += record_to_sample_from_dataset[dataset_name]

        while sampled_records < records_to_sample:
            increased = False
            for dataset_name in datasets:
                if dataset_name != target_dataset:
                    if record_to_sample_from_dataset[dataset_name] < record_per_dataset[
                            dataset_name] and sampled_records < records_to_sample:
                        record_to_sample_from_dataset[dataset_name] += 1
                        sampled_records += 1
                        increased = True
            if not increased:
                raise ValueError('Not enough records for training')

        for dataset_name in datasets:
            setting = settings[dataset_name]
            records_in_dataset = json.load(
                open(f"scripts/minimal_exemple/config/perf_by_records/" f"{dataset_name}.json")
            )
            df = pd.DataFrame.from_dict(records_in_dataset, orient="index")
            df.columns = ["score"]
            df = df.sort_values("score", ascending=False)

            output_folders = {
                record.replace(".h5", ""): setting["h5_directory"] + record
                for record in os.listdir(setting["h5_directory"])
            }
            records_in_dataset = list(output_folders.values())
            records_in_dataset.sort()
            dataset_memmap_hash = memmap_hash(memmaps_description[dataset_name])
            if not force:
                if not os.path.exists(setting["memmap_directory"]):
                    os.makedirs(setting["memmap_directory"])
                try:
                    already_created_record = os.listdir(
                        setting["memmap_directory"] + dataset_memmap_hash + "/"
                    )

                except FileNotFoundError:
                    pass

            if dataset_name in datasets:
                (memmaps_directory, groups_description, features_description,) = h5_to_memmaps(
                    records=records_in_dataset,
                    memmap_description=memmaps_description[dataset_name],
                    memmap_directory=setting["memmap_directory"],
                    num_workers=num_workers,
                    error_tolerant=True,
                )
            else:
                memmaps_directory = setting["memmap_directory"] + "/" + dataset_memmap_hash + "/"

            if dataset_name != target_dataset:
                records_for_train_and_validation = list(
                    df.index[:record_to_sample_from_dataset[dataset_name]])
                records_for_train_and_validation = [
                    x
                    for x in records_for_train_and_validation
                    if x in os.listdir(memmaps_directory)
                ]
                records_for_train_and_validation = [
                    memmaps_directory + record + "/"
                    for record in records_for_train_and_validation
                    if "." not in record
                ]
                datasets_split[dataset_name] = train_test_val_split(
                    records_for_train_and_validation, 0.7, 0.0, 0.3, seed=DATASET_SPLIT_SEED,
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

        results = {source_dataset: {} for source_dataset in source_datasets}
        train_split, validation_split, test_split = [], [], []
        for dataset in datasets:
            train_split += datasets_split[dataset]["train"]
            validation_split += datasets_split[dataset]["validation"]
            test_split += datasets_split[dataset]["test"]

        experiment_to_log = {
            "memmap_description": memmaps_description[target_dataset],
            "dataset_settings": settings[target_dataset],
            "trainer_parameters": trainer_parameters,
            "normalization_parameters": model_normalization,
            "net_parameters": model_description,
            "dataset_parameters": {
                "temporal_context": TEMPORAL_CONTEXT,
                "transform_parameters": [{"name": "eeg", "processing": []}],
                "split": {
                    "train": train_split,
                    "val": validation_split,
                    "test": test_split,
                    "parameters_init": train_split,
                },
            },
            "save_folder": EXPERIMENT_OUTPUT_FOLDER,
        }

        folder = log_experiment(
            **experiment_to_log, num_workers=num_workers, generate_memmaps=False
        )
