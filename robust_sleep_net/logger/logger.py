import hashlib
import json
import os
import shutil
import time
import uuid
from itertools import chain, combinations

import git

from ..datasets.dataset import DreemDataset
from ..models.modulo_net.net import ModuloNet
from ..models.modulo_net.normalization import initialize_standardization_parameters
from ..preprocessings.h5_to_memmap import h5_to_memmaps
from ..trainers import Trainer
from ..utils.train_test_val_split import train_test_val_split


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return [
        x for x in chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)) if len(x) >= 1
    ]


def memmap_hash(memmap_description):
    return hashlib.sha1(json.dumps(memmap_description).encode()).hexdigest()[:10]


def log_experiment(
    dataset_settings,
    memmap_description,
    dataset_parameters,
    normalization_parameters,
    trainer_parameters,
    net_parameters=None,
    save_folder=None,
    checkpoint=None,
    num_workers=0,
    experiment_id=None,
    generate_memmaps=True,
    validate_with_ablation_modalities=None,
    init_blacklist=[],
):
    if experiment_id is None:
        experiment_id = str(uuid.uuid4())
    if "num_workers" not in trainer_parameters:
        trainer_parameters["num_workers"] = num_workers

    if net_parameters == checkpoint:
        assert (
            net_parameters != None
        ), "Either the net parameters or an experiment to preload have to be provided"

    save_folder = save_folder + "/" + experiment_id + "/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    repo = git.Repo(search_parent_directories=True)
    metadata = {
        "git_branch": repo.active_branch.name,
        "git_hash": repo.head.object.hexsha,
        "begin": int(time.time()),
        "end": None,
        "experiment_id": experiment_id,
    }
    if "records_name" not in dataset_settings:
        records = [
            dataset_settings["h5_directory"] + record
            for record in os.listdir(dataset_settings["h5_directory"])
        ]
    else:
        records = [
            dataset_settings["h5_directory"] + record for record in dataset_settings["records_name"]
        ]

    if generate_memmaps:
        memmaps_directory, groups_description, features_description = h5_to_memmaps(
            records=records,
            memmap_description=memmap_description,
            memmap_directory=dataset_settings["memmap_directory"],
            num_workers=num_workers,
        )

        memmap_records = [
            memmaps_directory + record + "/"
            for record in os.listdir(memmaps_directory)
            if "." not in record
        ]

    else:
        memmaps_directory = (
            dataset_settings["memmap_directory"] + memmap_hash(memmap_description) + "/"
        )
        groups_description = json.load(open(memmaps_directory + "groups_description.json", "r"))
        features_description = json.load(open(memmaps_directory + "features_description.json", "r"))

    if isinstance(dataset_parameters["split"]["train"], list):
        train_records, test_records, validation_records = (
            dataset_parameters["split"]["train"],
            dataset_parameters["split"]["test"],
            dataset_parameters["split"]["val"],
        )
    else:
        train_records, test_records, validation_records = train_test_val_split(
            memmap_records, **dataset_parameters["split"]
        )
    print("Training records:", len(train_records))
    print("Validation records:", len(validation_records))
    if test_records is not None:
        for i, record in enumerate(test_records):
            assert record not in train_records
            assert record not in validation_records
        print("Test records:", len(test_records))

    dataset_train = DreemDataset(
        groups_description,
        features_description=features_description,
        transform_parameters=dataset_parameters["transform_parameters"],
        temporal_context=dataset_parameters["temporal_context"],
        hypnogram_filename=dataset_parameters.get('hypnogram_filename','hypno.mm'),
        records=train_records,
    )

    dataset_validation = DreemDataset(
        dataset_train.groups_description,
        features_description=features_description,
        temporal_context=dataset_parameters["temporal_context"],
        hypnogram_filename=dataset_parameters.get('hypnogram_filename', 'hypno.mm'),
        records=validation_records,
    )

    experiment_description = {
        "metadata": metadata,
        "dataset_settings": dataset_settings,
        "memmap_description": memmap_description,
        "groups_description": groups_description,
        "dataset_parameters": dataset_parameters,
        "normalization_parameters": normalization_parameters,
        "trainers_parameters": trainer_parameters,
        "net_parameters": net_parameters,
        "performance_on_test_set": None,
        "performance_on_test_set_online": None,
        "performance_per_records": None,
        "performance_per_records_online": None,
        "records_split": None,
    }
    json.dump(experiment_description, open(save_folder + "description.json", "w"), indent=4)

    if checkpoint is not None:
        assert (
            "directory" in checkpoint
        ), "The directory of the experiment to load has to be provided"
        assert "net_to_load" in checkpoint, "The net to load has to be provided"
        shutil.copytree(checkpoint["directory"], save_folder + "base_experiment/")
        net = ModuloNet.load(checkpoint["directory"] + checkpoint["net_to_load"])

        if "trainable_layers" in checkpoint:
            for name, param in net.named_parameters():
                if name not in checkpoint["trainable_layers"]:
                    print("Freezing: ", name)
                    param.requires_grad = False

        with open(checkpoint["directory"] + "description.json", "r") as desc_json:
            net_parameters = json.load(desc_json)["net_parameters"]
        print("Load net with parameters:", net_parameters)

    else:
        normalization_parameters_init = initialize_standardization_parameters(
            dataset_train, normalization_parameters, blacklist=init_blacklist
        )
        net = ModuloNet(
            groups=dataset_train.groups_description,
            features=dataset_train.features_description,
            normalization_parameters=normalization_parameters_init,
            net_parameters=net_parameters,
        )

    trainer_save_folder = save_folder + "training/"
    if not os.path.exists(trainer_save_folder):
        os.makedirs(trainer_save_folder)

    trainer = Trainer(net=net, save_folder=trainer_save_folder, **trainer_parameters["args"])

    trainer.train(train_dataset=dataset_train, validation_dataset=dataset_validation)

    metadata["end"] = int(time.time())
    net = ModuloNet.load(trainer_save_folder + "best_net")
    trainer = Trainer(net=net, save_folder=trainer_save_folder, **trainer_parameters["args"])
    modalities_in_train_set = set(dataset_train.groups_description.keys())
    if test_records is not None and len(test_records) > 0:
        dataset_test = DreemDataset(
            dataset_train.groups_description,
            features_description=features_description,
            temporal_context=dataset_parameters["temporal_context"],
            records=test_records,
            hypnogram_filename=dataset_parameters.get('hypnogram_filename_test', 'hypno.mm'),
        )

        del dataset_train
        del dataset_validation
        modalities_in_test_set = set(dataset_test.groups_description.keys())
        available_modalities = modalities_in_test_set.intersection(modalities_in_train_set)
        result_with_modalities_ablations = []
        if validate_with_ablation_modalities is not None:
            if validate_with_ablation_modalities == "full":
                modalities_for_validation = powerset(available_modalities)
            else:
                assert isinstance(validate_with_ablation_modalities, list)
                for x in validate_with_ablation_modalities:
                    for modality in x:
                        assert modality in available_modalities or modality == 'name', (
                            f"{x} is not in the modalities of " f"the test or " f"training dataset"
                        )
                modalities_for_validation = validate_with_ablation_modalities

            for i, modality in enumerate(modalities_for_validation):
                ablation_name = modality.get('name',i)
                modality = {k:v for k,v in modality.items() if k!= 'name'}

                if not os.path.exists(save_folder + "/hypnograms_ablation/"):
                    os.makedirs(save_folder + "/hypnograms_ablation/")
                (
                    performance_on_test_set,
                    _,
                    performance_per_records,
                    hypnograms,
                ) = trainer.validate(
                    dataset_test, return_metrics_per_records=True, modality_to_use=modality,
                )
                json.dump(
                    hypnograms,
                    open(f"{save_folder}/hypnograms_ablation/ablation_{ablation_name}.json", "w"),
                    indent=4,
                )
                performance_per_records = {
                    record.split("/")[-2]: metric
                    for record, metric in performance_per_records.items()
                }
                result_with_modalities_ablations += [
                    {
                        "modalities": modality,
                        "name": ablation_name,
                        "performance_on_test_set": performance_on_test_set,
                        "performance_per_records": performance_per_records,
                    }
                ]
            # results
            json.dump(
                result_with_modalities_ablations,
                open(save_folder + "modalities_ablation.json", "w"),
                indent=4,
            )

        (performance_on_test_set, _, performance_per_records, hypnograms,) = trainer.validate(
            dataset_test, return_metrics_per_records=True
        )
        performance_per_records = {
            record.split("/")[-2]: metric for record, metric in performance_per_records.items()
        }
        records_split = {
            "train_records": [record.split("/")[-2] for record in train_records],
            "validation_records": [record.split("/")[-2] for record in validation_records],
            "test_records": [record.split("/")[-2] for record in test_records],
        }
    else:
        performance_per_records = {}
        hypnograms = {}
        performance_on_test_set = {}
        records_split = {
            "train_records": [record.split("/")[-2] for record in train_records],
            "validation_records": [record.split("/")[-2] for record in validation_records],
            "test_records": [],
        }
        dataset_test = None

    # experiment_description
    experiment_description = {
        "metadata": metadata,
        "dataset_settings": dataset_settings,
        "memmap_description": memmap_description,
        "groups_description": groups_description,
        "features_description": features_description,
        "dataset_parameters": dataset_parameters,
        "normalization_parameters": normalization_parameters,
        "trainers_parameters": trainer_parameters,
        "net_parameters": net_parameters,
        "performance_on_test_set": performance_on_test_set,
        "performance_per_records": performance_per_records,
        "records_split": records_split,
    }

    # dump description
    json.dump(experiment_description, open(save_folder + "description.json", "w"), indent=4)

    json.dump(hypnograms, open(save_folder + "hypnograms.json", "w"), indent=4)

    net.save(save_folder + "best_model.gz")
    del dataset_test
    del trainer
    del net
    return save_folder
