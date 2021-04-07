import hashlib
import json
import os
import random as rd
import shutil

from robust_sleep_net.logger.logger import log_experiment
from robust_sleep_net.preprocessings.h5_to_memmap import h5_to_memmaps


def memmap_hash(memmap_description):
    return hashlib.sha1(json.dumps(memmap_description).encode()).hexdigest()[:10]


def split_list(a, n):
    k, m = divmod(len(a), n)
    return list(a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def run_experiment(
    settings,
    memmaps_description,
    temporal_context,
    trainer,
    normalization,
    model,
    split,
    transform,
    save_folder,
    hypnogram_filename_train = 'hypno.mm',
    hypnogram_filename_test ='hypno.mm',
    fold_to_run=None,
    force=False,
    error_tolerant=False,
    checkpoint=None,
    num_workers=0,
    SEED=2019,
    validate_with_ablation_modalities = None,
    max_training_records = None
):
    if os.path.exists(save_folder) and force:
        shutil.rmtree(save_folder)

    description_hash = memmap_hash(memmaps_description)
    print(settings["memmap_directory"])
    h5_to_memmaps(
        records=[
            settings["h5_directory"] + record for record in os.listdir(settings["h5_directory"])
        ],
        memmap_description=memmaps_description,
        memmap_directory=settings["memmap_directory"],
        error_tolerant=error_tolerant,
    )
    dataset_dir = settings["memmap_directory"] + description_hash + "/"
    available_dreem_records = [
        dataset_dir + record + "/" for record in os.listdir(dataset_dir) if ".json" not in record
    ]

    # build the folds
    rd.seed(SEED)
    rd.shuffle(available_dreem_records)

    assert split["type"] in ["loov", "kfolds"]

    if split["type"] == "kfolds":
        N_FOLDS = split["args"]["n_folds"]
        if "subjects" not in split["args"]:  # assumer record-wise split
            folds = split_list(available_dreem_records, N_FOLDS)
        else:  # assume multiple record per subject and subject-wise split
            subjects = []
            for subject in split["args"]["subjects"]:
                for record in subject["records"]:
                    if record in os.listdir(dataset_dir):
                        subjects += [subject]
                        break

            subject_per_folds = split_list(subjects, N_FOLDS)

            folds = []
            for subjects in subject_per_folds:
                record_in_fold = []
                for subject in subjects:
                    for record in subject["records"]:
                        record_in_fold += [dataset_dir + record + "/"]
                folds += [record_in_fold]

    elif split["type"] == "loov":
        # LOOV training
        folds = [[record] for record in available_dreem_records]
    else:
        raise ValueError

    if fold_to_run is None:
        fold_to_run = [j for j, _ in enumerate(folds)]

    outfolders = []
    for i, fold in enumerate(folds):

        if i in fold_to_run:
            other_folds = [fold for k, fold in enumerate(folds) if k != i]
            rd.seed(SEED + i)
            rd.shuffle(other_folds)
            n_val = max(1, int(len(other_folds) * 0.2))

            train_folds, val_folds = other_folds[n_val:], other_folds[:n_val]
            train_records = [record for train_fold in train_folds for record in train_fold]
            val_records = [record for val_fold in val_folds for record in val_fold]
            if max_training_records is not None:
                rd.shuffle(train_records)
                rd.shuffle(val_records)
                record_to_use_for_training = min(int(max_training_records * 0.7),
                                                 max_training_records - 1)
                record_to_use_for_validation = max_training_records - record_to_use_for_training
                train_records = train_records[:record_to_use_for_training]
                val_records = val_records[:record_to_use_for_validation]

            experiment_description = {
                "memmap_description": memmaps_description,
                "dataset_settings": settings,
                "trainer_parameters": trainer,
                "normalization_parameters": normalization,
                "net_parameters": model,
                "dataset_parameters": {
                    "split": {"train": train_records, "val": val_records, "test": fold},
                    "temporal_context": temporal_context,
                    "transform_parameters": transform,
                    "hypnogram_filename":hypnogram_filename_train,
                    "hypnogram_filename_test":hypnogram_filename_test
                },
                "save_folder": f"{save_folder}",
            }

            outfolders += [log_experiment(
                **experiment_description,
                num_workers=num_workers,
                generate_memmaps=False,
                checkpoint=checkpoint,
                validate_with_ablation_modalities=validate_with_ablation_modalities
            )]

    return outfolders
