import json
import os

import h5py
import numpy as np
import pyedflib
from xml.dom import minidom

from robust_sleep_net.utils.utils import standardize_signals_durations

sleep_stage_lookup = {
    "Wake|0": 0,
    "Stage 1 sleep|1": 1,
    "Stage 2 sleep|2": 2,
    "Stage 3 sleep|3": 3,
    "REM sleep|5": 4,
    "Unsure|Unsure": -1,
}

signal_groups = {
    "EOG": ["EOG(L)", "EOG(R)"],
    "EEG": ["EEG", "EEG(sec)"],
    "ECG": ["ECG"],
    "EMG": ["EMG"],
    "Respiratory": ["THOR RES", "ABDO RES", "AIRFLOW"],
}


def get_events(annotation_file):
    xmldoc = minidom.parse(annotation_file)
    events = []
    itemlist = xmldoc.getElementsByTagName("ScoredEvent")
    for s in itemlist:
        name = s.getElementsByTagName("EventConcept")[0].firstChild.data
        duration = int(float(s.getElementsByTagName("Duration")[0].firstChild.data))
        start = int(float(s.getElementsByTagName("Start")[0].firstChild.data))
        events += [{"name": name, "duration": duration, "start": start, "end": start + duration,}]
    return events


def build_hypnogram(events):
    record_start_time = 0
    record_end_time = max(event["end"] for event in events)
    hypnogram = np.zeros(record_end_time - record_start_time) - 1
    for event in events:
        if event["name"] in sleep_stage_lookup:
            hypnogram[event["start"] : event["end"]] = sleep_stage_lookup[event["name"]]

    return hypnogram[::30], record_start_time, record_end_time


def get_annotation(events, annotation_name, sampling_freq=64):
    """
    Extract annotation from an EDF file
    :param annotation_file : EDF handle
    :param annotation_name :(str) name of the annoation to get
    :param sampling_freq : (int) sampling freq to use to build the event binary representation
    :return:
    """

    record_duration = max(event["end"] for event in events)
    result = np.zeros(record_duration * sampling_freq)
    time_begins, durations = [], []
    for event in events:
        if event["name"] == annotation_name:
            time_begins += [event["start"]]
            durations += [event["duration"]]
            time_begin = int(event["start"] * sampling_freq)
            time_end = int(event["end"] * sampling_freq)
            result[time_begin:time_end] = 1
    return result, time_begins, durations


def to_h5(record_file, annotation_file, h5_target_directory, crop_record=True):
    """
    Format a MASS EDF record and its annotation to a standardized h5
    :param record_file :(str)
    :param annotation_files :(list of str) the hypnogram has to be in the first annotation file
    :param output_file :(str)
    :param crop_record : (bool)
    :return:
    """
    description = []
    with pyedflib.EdfReader(record_file) as data:
        with h5py.File(h5_target_directory, "w", driver="core") as h5_target:
            signal_labels = {key: value for value, key in enumerate(data.getSignalLabels())}

            annotation_file = get_events(annotation_file)
            hypno, time_begin, time_end = build_hypnogram(annotation_file)
            h5_target["hypnogram"] = np.array(hypno).astype(int)

            # Add signal
            h5_target.create_group("signals")
            for group_name, signals_list in signal_groups.items():
                h5_target["signals"].create_group(group_name)
                mod_fs = None
                mod_unit = None
                for signal in signals_list:
                    signal_path = "signals/" + group_name + "/" + signal
                    signal_idx = signal_labels[signal]
                    if mod_fs is None:
                        mod_fs = int(data.getSignalHeader(signal_idx)["sample_rate"])
                        mod_unit = data.getSignalHeader(signal_idx)["dimension"]
                    if mod_fs is not None:
                        if (
                            mod_fs == data.getSignalHeader(signal_idx)["sample_rate"]
                            and mod_unit == data.getSignalHeader(signal_idx)["dimension"]
                        ):
                            if crop_record:
                                begin_idx = int(time_begin * mod_fs)
                                end_idx = int(time_end * mod_fs)
                                h5_target.create_dataset(
                                    signal_path,
                                    data=data.readSignal(signal_idx)[begin_idx:end_idx],
                                    compression="gzip"
                                )
                            else:
                                h5_target[group_name][signal] = data.readSignal(signal_idx)
                            signal_description = {
                                "fs": mod_fs,
                                "unit": mod_unit,
                                "path": signal_path,
                                "name": signal,
                                "domain": group_name,
                                "default": True,
                                "default": True
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

            h5_target.attrs.create("description", json.dumps(description), dtype=np.dtype("S32768"))
            h5_target.attrs.create("events_description", json.dumps([]), dtype=np.dtype("S32768"))

            # truncate file
            h5_target.attrs["duration"] = standardize_signals_durations(h5_target)

            h5_target.close()
            print("Sucess: ", h5_target_directory)
            return True


if __name__ == "__main__":

    from scripts.settings import SHHS_SETTINGS
    from joblib import Parallel, delayed

    records_directory, annotations_directory, h5_directory = (
        SHHS_SETTINGS["records_directory"],
        SHHS_SETTINGS["annotations_directory"],
        SHHS_SETTINGS["h5_directory"],
    )

    if not os.path.exists(h5_directory):
        os.mkdir(h5_directory)

    records = os.listdir(records_directory)
    annotations = os.listdir(annotations_directory)
    records.sort()
    annotations.sort()
    assert len(annotations) <= len(records)

    parallel = False

    if parallel is True:
        Parallel(n_jobs=-1)(
            delayed(to_h5)(
                records[i], annotations[i], (h5_directory + records[i]).replace(".edf", ".h5"),
            )
            for i in range(len(records))
        )
    else:
        for i in range(len(records)):
            if not os.path.exists((h5_directory + records[i]).replace(".edf", ".h5")):
                to_h5(
                    records_directory + records[i],
                    annotations_directory + annotations[i],
                    (h5_directory + records[i]).replace(".edf", ".h5"),
                )
