import json

import h5py
import mne
import numpy as np
import pandas as pd
from dateutil import parser
from scipy import interpolate
import os
from robust_sleep_net.utils.utils import standardize_signals_durations

correspondance = {
    "O1A2": "O1-A2",
    "O2A1": "O2-A1",
    "F4A1": "F4-A1",
    "F3A2": "F3-A2",
    "C3A2": "C3-A2",
    "C4A1": "C4-A1"
}


def resample(signal, base_fs, target_fs, kind="cubic"):
    signal_frequency = base_fs
    resampling_ratio = signal_frequency / target_fs
    x_base = np.arange(0, len(signal))
    interpolator = interpolate.interp1d(
        x_base, signal, axis=0, bounds_error=False, fill_value="extrapolate", kind=kind
    )

    x_interp = np.arange(0, len(signal), resampling_ratio)
    resampled_signal = interpolator(x_interp)
    return resampled_signal


sleep_staging_events = [
    "SLEEP-REM",
    "SLEEP-S0",
    "SLEEP-S1",
    "SLEEP-S2",
    "SLEEP-S3",
    "SLEEP-S4",
    "SLEEP-UNSCORED",
]

stages_lookup = {
    "SLEEP-S1": 1,
    "SLEEP-S2": 2,
    "SLEEP-S3": 3,
    "SLEEP-S4": 3,
    "SLEEP-UNSCORED": -1,
    "SLEEP-REM": 4,
    "SLEEP-S0": 0,
}


def diff_times_in_seconds(t1, t2):
    # caveat emptor - assumes t1 & t2 are python times, on the same day and t2 is after t1
    h1, m1, s1 = t1.hour, t1.minute, t1.second
    h2, m2, s2 = t2.hour, t2.minute, t2.second
    if h1 - h2 > 10:
        h2 += 24

    t1_secs = s1 + 60 * (m1 + 60 * h1)
    t2_secs = s2 + 60 * (m2 + 60 * h2)
    return t2_secs - t1_secs


def get_sleep_stages(staging_file, psg_start_time):
    for i in range(15, 25):
        try:
            annotations = pd.read_csv(staging_file, sep="\t", skiprows=range(i))

            if "Event" in annotations.columns:
                break
        except:
            pass
    x = annotations[["Time [hh:mm:ss]", "Event"]]
    staging_events, sleep_stages = [], []
    for i, row in x.iterrows():
        d = row.to_dict()
        if d["Event"] in sleep_staging_events:
            staging_events += [d]
            sleep_stages += [stages_lookup[d["Event"]]]
    start_time = staging_events[0]["Time [hh:mm:ss]"].replace(".", ":")
    staging_start = parser.parse(start_time).time()
    truncation_start = diff_times_in_seconds(psg_start_time, staging_start)
    if truncation_start < 0:
        epoch_to_truncate = int(abs(truncation_start) // 30 + 1)
        truncation_start += epoch_to_truncate * 30
        sleep_stages = sleep_stages[epoch_to_truncate:]
    assert truncation_start >= 0
    duration = truncation_start + 30 * len(sleep_stages)
    assert duration > 0
    return sleep_stages, truncation_start, duration


def to_h5(record_file, staging_file, h5_target_directory, signals, force=False):
    """
    Format a MASS EDF record and its annotation to a standardized h5
    record_file :(str)
    annotation_files :(list of str) the hypnogram has to be in the first annotation file
    h5_target :(str)
    crop_record : (bool)
    """
    description = []
    events_description = []
    with mne.io.read_raw_edf(record_file) as data:
        if force or not os.path.exists(h5_target_directory):
            with h5py.File(h5_target_directory, "w", driver="core") as h5_target:
                signal_labels = {key: value for value, key in enumerate(data.ch_names)}
                psg_start_time = data.info["meas_date"]

                hypno, time_begin, time_end = get_sleep_stages(staging_file, psg_start_time)
                h5_target["hypnogram"] = np.array(hypno).astype(int)

                # Add signal
                h5_target.create_group("signals")
                for signal in signals:
                    if (
                        isinstance(signal, tuple)
                        and signal[0] in signal_labels
                        and signal[1] in signal_labels
                    ):
                        ref, base = signal[1], signal[0]
                        ref_data, ref_t = data[ref]
                        ref_data = ref_data.reshape(-1)
                        ref_fs = int(1 / (ref_t[1] - ref_t[0]))

                        base_data, base_t = data[base]
                        base_data = base_data.reshape(-1)
                        base_fs = int(1 / (base_t[1] - base_t[0]))
                        if base_fs == ref_fs:
                            x = ref_data - base_data

                        mod_fs = ref_fs
                        mod_unit = data._orig_units[ref]
                        x = x.astype(np.float32)
                        sig_name = ref + "-" + base
                    elif signal in signal_labels:
                        x, t = data[signal]
                        x = x.reshape(-1)
                        mod_fs = int(1 / (t[1] - t[0]))
                        mod_unit = data._orig_units[signal]
                        sig_name = correspondance.get(signal, signal)
                    else:
                        continue

                    signal_path = "signals/" + sig_name

                    begin_idx = int(time_begin * mod_fs)
                    end_idx = int(time_end * mod_fs)
                    x = x[begin_idx:end_idx]
                    h5_target.create_dataset(signal_path, data=x, compression="gzip")
                    signal_description = {
                        "fs": mod_fs,
                        "unit": mod_unit,
                        "path": signal_path,
                        "name": sig_name,
                        "domain": sig_name,
                        "default": True,
                    }
                    description += [signal_description]
                    h5_target["signals/" + sig_name].attrs["fs"] = mod_fs
                    h5_target["signals/" + sig_name].attrs["unit"] = mod_unit

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


def format_cap_to_h5(settings,parallel):
    folder = settings["base_directory"]
    signals = {}
    records = {}
    for file in os.listdir(settings["base_directory"]):
        try:
            if ".edf" in file:
                psg = mne.io.read_raw_edf(f"{folder}/{file}", verbose=-1)
                record_name = file.replace(".edf", "")
                sigs = psg.info["ch_names"]
                signals[file] = []
                for sig in sigs:
                    for elt in [
                        "EMG",
                        "ECG",
                        "C4",
                        "C3",
                        "A1",
                        "A2",
                        "F3",
                        "F4",
                        "EOG",
                        "O1",
                        "O2",
                        "ROC",
                        "LOC",
                    ]:
                        if elt in sig and sig not in ["SAO2", "SpO2"]:
                            signals[file] += [sig]
                signals[file] = set(signals[file])
                if os.path.exists(f"{folder}/{record_name}.txt"):
                    records[record_name] = (
                        f"{folder}/{file}",
                        f"{folder}/{record_name}.txt",
                    )
                psg.close()
        except NotImplementedError:
            if ".edf" in file and ".st" not in file:
                print(file)

    available_signals = set()
    for x in signals.values():
        available_signals = available_signals.union(set(x))
    for elt in available_signals:
        if "-" in elt:
            base, reference = elt.split("-")
            available_signals = available_signals.union([(base, reference)])

    available_signals = set(
        x for x in available_signals if "-" in x or isinstance(x, tuple) or x in correspondance
    )

    h5_folder = settings["h5_directory"]
    if parallel is True:
        Parallel(n_jobs=-1)(
            delayed(to_h5)(x[0], x[1], f"{h5_folder}/{record}.h5", available_signals)
            for record, x in records.items()
        )
    else:
        for record, x in records.items():
            to_h5(x[0], x[1], f"{h5_folder}/{record}.h5", available_signals)


if __name__ == "__main__":
    from joblib import Parallel, delayed
    from scripts.settings import CAP_SETTINGS
    format_cap_to_h5(CAP_SETTINGS,False)