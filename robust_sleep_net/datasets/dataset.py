"""
dataset
"""
import copy
import json
import random as rd

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_augmentation import augment_data


class DreemDataset(Dataset):
    def __init__(
        self,
        groups_description,
        features_description=None,
        transform_parameters=None,
        temporal_context=1,
        records=None,
        hypnogram_filename = 'hypno.mm'
    ):
        if not isinstance(transform_parameters, list) and transform_parameters is not None:
            raise TypeError("transform_parameters should be a list")

        self.hypnogram = {}
        self.data = {}
        self.features_data = {}
        self.idx_to_record = []
        self.idx_to_record_eval = []
        self.temporal_context = temporal_context
        self.input_temporal_dimension = 1
        self.hypnogram_filename = hypnogram_filename
        assert (
            self.temporal_context == 1 or self.input_temporal_dimension == 1
        ), "Either temporal context or input temporal context should be set to one."
        self.max_temporal_context = temporal_context
        self.records = []
        self.groups = list(groups_description.keys())
        self.groups_description = copy.deepcopy(groups_description)
        self.channels_per_group = {k: v["shape"][-1] for k, v in groups_description.items()}
        self.channels_per_group_to_sample = copy.deepcopy(self.channels_per_group)
        self.channels_per_group_in_record = {}
        self.channels_per_group_frequency = {k: [] for k in groups_description}
        self.sampling_probability_channels = {}
        self.max_number_of_channels = {}

        self.features = list(features_description.keys())
        self.features_description = features_description
        self.transform_parameters = transform_parameters
        self.record_index = {}
        self.record_index_eval = {}
        self.target_frequency = {}

        if self.transform_parameters is not None:
            for group in self.groups:
                assert group in [
                    group["name"] for group in transform_parameters
                ], "augmentation pipeline is invalid"

        if records is not None:
            for record in records:
                self.add_record(record)
        self.compute_channels_probabilities()

    def compute_channels_probabilities(self):
        for group in self.channels_per_group_frequency:
            n_channel, count = np.unique(
                self.channels_per_group_frequency[group], return_counts=True
            )
            self.sampling_probability_channels[group] = n_channel, count / np.sum(count)
            self.max_number_of_channels[group] = np.max(self.channels_per_group_frequency[group])

    def get_number_of_channels_to_sample_per_group(self, channels_to_sample=None):
        if isinstance(channels_to_sample, dict):
            return channels_to_sample
        else:
            channels_to_sample = {}
            for group in self.channels_per_group_frequency:
                if len(self.channels_per_group_frequency[group]) > 0:
                    max_number_of_channels = self.max_number_of_channels[group]
                    if max_number_of_channels > 1:
                        number_of_channels = np.random.choice(np.arange(1, max_number_of_channels))
                    else:
                        number_of_channels = 1
                    channels_to_sample[group] = np.random.choice(
                        np.arange(self.max_number_of_channels[group]),
                        size=number_of_channels,
                        replace=False,
                    )
            return channels_to_sample

    def get_record(
        self,
        record,
        batch_size=64,
        return_index=False,
        mode="train",
        stride=1,
        modality_to_use=None,
    ):
        if mode == "train":
            index_min, index_max = self.record_index[record]
        else:
            index_min, index_max = self.record_index_eval[record]
        number_of_samples = index_max - index_min
        batch_size = min(batch_size, number_of_samples // 3)
        for i in range(number_of_samples // batch_size + 1):
            element_in_batch = 0
            batch_data = {}
            batch_data["groups"] = {}
            batch_data["features"] = {}
            batch_data["hypnogram"] = []
            indexes = []
            for j in range(i * batch_size * stride, (i + 1) * batch_size * stride, stride):
                if j + index_min <= index_max:
                    data = self.__getitem__(j + index_min, mode=mode)
                    batch_data["hypnogram"] += [data["hypnogram"].unsqueeze(0)]
                    for group in data["groups"]:
                        select_channels = isinstance(modality_to_use, dict)
                        if modality_to_use is None or group in modality_to_use:
                            if group not in batch_data["groups"]:
                                batch_data["groups"][group] = []
                            if select_channels:
                                channels_to_use = np.array(modality_to_use[group])
                            else:
                                channels_to_use = np.arange(
                                    0, self.channels_per_group_in_record[record][group]
                                )

                            batch_data["groups"][group] += [
                                data["groups"][group][:, channels_to_use].unsqueeze(0)
                            ]
                    for feature in data["features"]:
                        if feature not in batch_data["features"]:
                            batch_data["features"][feature] = []

                        batch_data["features"][feature] += [data["features"][feature].unsqueeze(0)]
                    element_in_batch += 1
                    if mode == "train":
                        indexes += [self.idx_to_record[j + index_min]["index"]]
                    else:
                        indexes += [self.idx_to_record_eval[j + index_min]["index"]]

            if element_in_batch > 0:
                for group in batch_data["groups"]:
                    batch_data["groups"][group] = torch.cat(batch_data["groups"][group])
                for feature in batch_data["features"]:
                    batch_data["features"][feature] = torch.cat(batch_data["features"][feature])

                batch_data["hypnogram"] = torch.cat(batch_data["hypnogram"])
                if return_index:
                    yield batch_data, indexes
                else:
                    yield batch_data

    def add_record(self, record):
        with open(record + "properties.json") as f:
            record_description = json.load(f)
        with open(record + "features_description.json") as f:
            features_description = json.load(f)

        window_length = max(self.temporal_context, self.input_temporal_dimension)
        self.records += [record]
        self.data[record] = {}
        self.features_data[record] = {}
        self.channels_per_group_in_record[record] = {}
        groups = record_description
        for group in groups:

            shape = tuple(groups[group]["shape"])
            self.data[record][group] = (record + "signals/" + group + ".mm", shape)
            self.channels_per_group_in_record[record][group] = shape[1]
            if group in self.channels_per_group_frequency:
                self.channels_per_group_frequency[group] += [
                    self.channels_per_group_in_record[record][group]
                ]

            else:
                self.channels_per_group_frequency[group] = [
                    self.channels_per_group_in_record[record][group]
                ]

        for feature in self.features_description:
            shape = tuple(features_description[feature]["shape"])
            self.features_data[record][feature] = (
                record + "features/" + feature + ".mm",
                shape,
            )


        self.hypnogram[record] = np.memmap(record + self.hypnogram_filename, mode="r", dtype="float32")

        target_frequency = np.unique(self.hypnogram[record], return_counts=True)
        for i, target in enumerate(target_frequency[0]):
            if target not in self.target_frequency:
                self.target_frequency[target] = target_frequency[1][i]
            else:
                self.target_frequency[target] += target_frequency[1][i]

        # Compute window for training
        valid_window = np.where(self.hypnogram[record] != -1)[0]

        valid_window = valid_window[valid_window >= window_length // 2]
        valid_window = valid_window[
            valid_window <= self.hypnogram[record].shape[0] - window_length // 2 - 1
        ]
        self.idx_to_record += [{"record": record, "index": index,} for index in valid_window]

        self.record_index[record] = [
            i
            for i, idx_to_record in enumerate(self.idx_to_record)
            if idx_to_record["record"] == record
        ]
        self.record_index[record] = (
            np.min(self.record_index[record]),
            np.max(self.record_index[record]),
        )
        # Compute window for evaluation (we never want to predict -1)
        valid_window_eval = np.arange(0, len(self.hypnogram[record]))
        valid_window_eval = valid_window_eval[valid_window_eval >= window_length // 2]
        valid_window_eval = valid_window_eval[
            valid_window_eval <= self.hypnogram[record].shape[0] - window_length // 2 - 1
        ]
        self.idx_to_record_eval += [
            {"record": record, "index": index,} for index in valid_window_eval
        ]

        self.record_index_eval[record] = [
            i
            for i, idx_to_record in enumerate(self.idx_to_record_eval)
            if idx_to_record["record"] == record
        ]
        self.record_index_eval[record] = (
            np.min(self.record_index_eval[record]),
            np.max(self.record_index_eval[record]),
        )

    def set_device(self, device):
        self.device = device

    def __len__(self):
        return len(self.idx_to_record) - 1

    def __getitem__(self, idx, mode="train"):
        sample = {}
        if mode == "train":
            record = self.idx_to_record[idx]["record"]
            idx = self.idx_to_record[idx]["index"]
            temporal_context = self.temporal_context
        elif mode == "eval":
            record = self.idx_to_record_eval[idx]["record"]
            idx = self.idx_to_record_eval[idx]["index"]
            temporal_context = self.max_temporal_context
        else:
            raise ValueError

        sample["record"] = record

        sample["groups"] = {}
        sample["features"] = {}

        # retrieve groups
        start_idx = idx - temporal_context // 2
        end_idx = idx + temporal_context // 2 + 1
        for group in self.data[record]:
            if mode == "train":
                channels_to_sample = np.random.choice(
                    np.arange(self.channels_per_group_in_record[record][group]),
                    size=self.max_number_of_channels[group],
                )

            else:
                channels_to_sample = np.arange(0, self.channels_per_group_in_record[record][group])

            window_length = self.groups_description[group]["window_length"]
            group_data = np.memmap(
                self.data[record][group][0],
                mode="r",
                dtype="float32",
                shape=self.data[record][group][1],
            )
            sample["groups"][group] = np.copy(
                group_data[start_idx * window_length : end_idx * window_length, channels_to_sample,]
            )
            group_data._mmap.close()
            shape = (temporal_context, window_length) + sample["groups"][group].shape[1:]
            sample["groups"][group] = sample["groups"][group].reshape(*shape)
            axis = (0,) + tuple(range(2, len(shape))) + (1,)
            sample["groups"][group] = np.transpose(sample["groups"][group], axis)

        # transform
        if self.transform_parameters is not None and mode == "train":
            sample["groups"] = augment_data(sample["groups"], self.transform_parameters)


        for group in sample["groups"]:
            sample["groups"][group] = torch.Tensor(sample["groups"][group])

        for feature in self.features:
            feature_data_mm = np.copy(
                np.memmap(
                    self.features_data[record][feature][0],
                    mode="r",
                    dtype="float32",
                    shape=self.features_data[record][feature][1],
                )
            )

            feature_data = np.copy(feature_data_mm)
            feature_data_mm._mmap.close()
            sample["features"][feature] = torch.Tensor(feature_data[start_idx:end_idx])

        # Retrieve hypnogram
        hypnogram = self.hypnogram[record][start_idx:end_idx]
        sample["hypnogram"] = torch.LongTensor(np.copy(hypnogram))
        return sample

    @staticmethod
    def load(serialized_dataset):
        pipeline = serialized_dataset["augmentation_pipeline"]
        window_length = serialized_dataset["temporal_context"]
        return DreemDataset(pipeline, window_length)
