import json

import h5py
import pyedflib
from robust_sleep_net.utils.utils import standardize_signals_durations

import mne
import numpy as np
import pandas as pd
from dateutil import parser

stages_lookup = {
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Movement time": -1,
    "Sleep stage ?": -1,
    "Sleep stage R": 4,
    "Sleep stage W": 0,
}


def compute_onset(lights_off, record_start):
    from datetime import datetime, date

    start, end = parser.parse(record_start).time(), parser.parse(lights_off).time()
    start, end = (
        datetime.combine(date.today(), start).timestamp(),
        datetime.combine(date.today(), end).timestamp(),
    )
    if end < start - 60 * 2:
        end += 60 * 60 * 24
    if end < start:
        end = start
    epoch_onset = end - start
    return epoch_onset


def get_sleep_stages_no_light(annotation_file, record_start, lights_off):
    """
    Extract the sleep stages from an annotation file
    annotation_file : (str) path to EDF annotation file
    returns stages: list of sleep stages
            time_begin: beginning of hypno
            time_end: end of hypno
    """
    annotation = mne.read_annotations(annotation_file)
    seconds_before_lights_off = compute_onset(lights_off, record_start)
    onsets = annotation.onset
    durations = annotation.duration
    tot_annotation_duration = onsets[-1] + durations[-1]
    tot_epoch = int(tot_annotation_duration // 30)
    stages = np.array([-1] * tot_epoch)
    labels = annotation.description

    for i, (onset, duration, label) in enumerate(zip(onsets, durations, labels)):
        start_epoch = int(onset // 30)
        end_epoch = int(start_epoch + duration // 30)
        stages[start_epoch:end_epoch] = stages_lookup.get(label, -1)
    stages_as_array = np.array(stages)
    first_sleep, last_sleep = (
        np.where(stages_as_array > 0)[0][0],
        np.where(stages_as_array > 0)[0][-1],
    )

    first_sleep, last_sleep = (
        max(0, first_sleep - 60),
        min(len(stages_as_array), last_sleep + 60),
    )
    lights_off, lights_on = (
        seconds_before_lights_off,
        min(len(stages_as_array), last_sleep + 30) * 30,
    )
    staging_start = lights_off // 30
    staging_end = lights_on // 30
    staging_start, staging_end = int(staging_start), int(staging_end)
    stages = stages[staging_start:staging_end].tolist()

    return stages, staging_start * 30, staging_end * 30


def get_annotation(annotation_file, annotation_name, sampling_freq=64):
    """
    Extract annotation from an EDF file
    annotation_file : EDF handle
    annotation_name :(str) name of the annoation to get
    sampling_freq : (int) sampling freq to use to build the event binary representation
    """

    annotations = annotation_file.readAnnotations()
    result = np.zeros(annotation_file.file_duration * sampling_freq)
    annot_idx = np.where(annotations[2] == annotation_name)[0]
    time_begins, durations = [], []
    for idx in annot_idx:
        time_begins += [annotations[0][idx]]
        durations += [annotations[1][idx]]
        time_begin = int(annotations[0][idx] * sampling_freq)
        time_end = time_begin + int(annotations[1][idx] * sampling_freq)
        result[time_begin:time_end] = 1
    return result, time_begins, durations


def to_h5(
    record_file,
    annotation_files,
    h5_target_directory,
    signals,
    record_start,
    lights_off,
    crop_record=True,
    force=False,
):
    """
    Format a MASS EDF record and its annotation to a standardized h5
    record_file :(str)
    annotation_files :(list of str) the hypnogram has to be in the first annotation file
    h5_target :(str)
    crop_record : (bool)
    """
    description = []
    events_description = []
    with pyedflib.EdfReader(record_file) as data:
        if force or not os.path.exists(h5_target_directory):
            with h5py.File(h5_target_directory, "w", driver="core") as h5_target:
                signal_labels = {key: value for value, key in enumerate(data.getSignalLabels())}

                hypno, time_begin, time_end = get_sleep_stages_no_light(
                    annotation_files, record_start=record_start, lights_off=lights_off
                )
                h5_target["hypnogram"] = np.array(hypno).astype(int)

                # Add signal
                h5_target.create_group("signals")
                for group_name, signals_list in signals.items():
                    group_name = group_name.lower()
                    h5_target["signals"].create_group(group_name)
                    mod_fs = None
                    mod_unit = None
                    for signal in signals_list:
                        signal_idx = signal_labels[signal]
                        if mod_fs is None:
                            mod_fs = int(data.getSignalHeader(signal_idx)["sample_rate"])
                            mod_unit = data.getSignalHeader(signal_idx)["dimension"]
                        if mod_fs is not None:
                            signal_path = "signals/" + group_name + "/" + signal
                            if (
                                mod_fs == data.getSignalHeader(signal_idx)["sample_rate"]
                                and mod_unit == data.getSignalHeader(signal_idx)["dimension"]
                            ):
                                if crop_record:
                                    begin_idx = int(time_begin * mod_fs)
                                    end_idx = int(time_end * mod_fs)
                                    x = data.readSignal(signal_idx)[begin_idx:end_idx].astype(
                                        np.float32
                                    )
                                    h5_target.create_dataset(
                                        signal_path, data=x, compression="gzip"
                                    )
                                else:
                                    x = data.readSignal(signal_idx).astype(np.float32)
                                    h5_target.create_dataset(
                                        signal_path, data=x, compression="gzip"
                                    )
                                signal_description = {
                                    "fs": mod_fs,
                                    "unit": mod_unit,
                                    "path": signal_path,
                                    "name": signal,
                                    "domain": group_name,
                                    "default": True,
                                }
                                description += [signal_description]
                            else:
                                print(
                                    "Signal: ",
                                    signal,
                                    "has invalid frequency or dimension for the modality",
                                )

                    h5_target["signals/" + group_name].attrs["fs"] = mod_fs
                    h5_target["signals/" + group_name].attrs["unit"] = mod_unit

                h5_target.attrs.create(
                    "description", json.dumps(description), dtype=np.dtype("S32768")
                )
                h5_target.attrs.create(
                    "events_description", json.dumps(events_description), dtype=np.dtype("S32768"),
                )

                # truncate file
                h5_target.attrs["duration"] = standardize_signals_durations(h5_target)

                h5_target.close()
                print("Sucess: ", h5_target_directory)
                return True
    return True


# List of MASS event

stages_lookup = {
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Movement time": -1,
    "Sleep stage ?": -1,
    "Sleep stage R": 4,
    "Sleep stage W": 0,
}

if __name__ == "__main__":
    import os
    from scripts.settings import BASE_DIRECTORY

    url = "https://physionet.org/files/sleep-edfx/1.0.0/"
    filename = os.system(f"wget -r -N -c -np -P {BASE_DIRECTORY} {url} ")

    from scripts.settings import SLEEP_EDF_IN_BED_SETTINGS
    from joblib import Parallel, delayed

    records_directory, h5_directory = (
        SLEEP_EDF_IN_BED_SETTINGS["edf_directory"],
        SLEEP_EDF_IN_BED_SETTINGS["h5_directory"],
    )
    records = {}
    for directory in records_directory:
        records.update({x: directory for x in os.listdir(directory) if "-PSG.edf" in x})

    meta_data = pd.read_csv("scripts/datasets_processing/sleep_edf/sleep_edf_meta.csv")

    lights_time = {}

    record_annotation = {}
    for record_name, directory in records.items():
        record_id = record_name[:7]
        record_meta_data = meta_data[meta_data["filename"] == record_name.replace(".edf", "")]
        lights_time[f"{directory}/{record_name}"] = (
            record_meta_data["start_time"].values[0],
            record_meta_data["LightsOff"].values[0],
        )

        annotations_in_directory = [
            x for x in os.listdir(directory) if "-Hypnogram.edf" in x and record_id in x
        ]

        assert len(annotations_in_directory) == 1, print(record_name, record_id)
        record_annotation[f"{directory}/{record_name}"] = (
            f"{directory}/" f"{annotations_in_directory[0]}"
        )

    if not os.path.exists(h5_directory):
        os.mkdir(h5_directory)

    parallel = False

    def record_to_h5_full(record, annotation, record_start, lights_off, force=False):
        with pyedflib.EdfReader(record) as f:
            channels = f.getSignalLabels()
        signals_prefix = ["EEG", "EMG", "ECG", "EOG"]
        signals = {}
        for channel in channels:
            for prefix in signals_prefix:
                if prefix in channel:
                    if prefix not in signals:
                        signals[prefix] = []
                    signals[prefix] += [channel]
                    break

        output_file = h5_directory + record.split("/")[-1].replace(".edf", ".h5")

        to_h5(
            record,
            annotation,
            output_file,
            record_start=record_start,
            lights_off=lights_off,
            signals=signals,
            force=force,
        )

    if parallel is True:
        Parallel(n_jobs=-1)(
            delayed(record_to_h5_full)(record, annotation, *lights_time[record])
            for record, annotation in record_annotation.items()
        )
    else:
        for record, annotation in record_annotation.items():
            record_to_h5_full(record, annotation, *lights_time[record], force=False)
