import os
import h5py
import json
import pyedflib
import pint
import numpy as np
from scipy.signal import resample_poly


def poly_resample(signal, source_fs, target_frequency):
    signal_frequency = source_fs
    if signal_frequency != target_frequency:
        signal_duration = signal.shape[0] / signal_frequency
        resampled_length = round(signal_duration * target_frequency)
        resampled_signal = resample_poly(signal, target_frequency, signal_frequency, axis=0)
        if len(resampled_signal) < resampled_length:
            padding = np.zeros((resampled_length - len(resampled_signal), signal.shape[-1]))
            resampled_signal = np.concatenate([resampled_signal, padding])
        return resampled_signal
    else:
        return signal


def return_scale_factor(unit):
    ureg = pint.UnitRegistry()
    try:
        signal_unit = ureg.parse_expression(unit)
        is_volt = any([elt == "volt" for elt in signal_unit.compatible_units()])
    except:
        is_volt = False

    if is_volt:
        factor = signal_unit.to("uV").magnitude
        return factor, "uV"
    else:
        return 1, unit


def clean_derivation(signal):
    '''for takeda "EEG O1-A1"'''
    clean_name = signal
    if signal[:4] == 'EEG ':
        clean_name = clean_name.split(' ')[1].split('-')[0]
    return clean_name


def match_derivations_from_edf(edf):
    matched_electrodes = []
    electrodes = [
        "F3",
        "F4",
        "F7",
        "F8",
        "Fz",
        "Fpz",
        "Fp1",
        "Fp2",
        "C3",
        "C4",
        "Cz",
        "A2",
        "A1",
        "M1",
        "M2",
        "O1",
        "O2",
        "Oz",
        "E1",
        "E2",
        "EEG",
        "EOG"
    ]
    electrodes += [x.lower() for x in electrodes]

    signals_in_edf = edf.getSignalHeaders()
    for signal in signals_in_edf:
        if signal['sample_rate'] >= 50 or signal['dimension'] in ['uV', 'mV']:
            for elt in electrodes:
                if elt in signal['label']:
                    matched_electrodes += [signal['label']]
    return list(set(matched_electrodes))


def build_derivation_setup_from_signal_headers(edf, electrodes=None):
    signals_in_edf = edf.getSignalHeaders()
    derivations = []
    signals_list = []
    for signal in signals_in_edf:
        signals_list += [signal['label']]
        if signal['label'] in electrodes:
            signal_name = signal["label"]
            derivations += [{"base": signal_name}]
    signals_list = set(signals_list)
    for elt in electrodes:
        if '-' in elt:
            elt = tuple(elt.split('-'))
        if isinstance(elt, tuple):
            if elt[0] in signals_list and elt[1] in signals_list:
                derivations += [{"base": elt[1], "reference": elt[0]}]

    return derivations


def is_record_compatible_with_setup(derivations, signal_index):
    missing_derivations = []
    for derivation in derivations:
        base = derivation["base"]
        if base not in signal_index:
            missing_derivations += [base]
        reference = derivation.get("reference", None)
        if reference is not None:
            if reference not in signal_index:
                missing_derivations += [reference]
    return missing_derivations == [], missing_derivations


def edf_to_h5(filename, force=False, h5_filename=None, electrodes=None, lights_on=False,
              lights_off=False, start_minute=False, start_30s=False
              ):
    """
    Conversion of EDF to h5 file
    """

    if h5_filename is None:
        h5_filename = filename[:-3] + "h5"

    if os.path.isfile(h5_filename) and force is False:
        return h5_filename

    h5 = h5py.File(h5_filename, "w")

    descriptor = []
    edf = pyedflib.EdfReader(filename)
    if electrodes is None:
        electrodes = match_derivations_from_edf(edf)
        assert len(
            electrodes) > 0, 'No matching electrodes were found for sleep staging, please provide suitable electrodes manually'

    signal_index = {edf.getLabel(i): i for i in range(edf.signals_in_file)}

    # We add global attributes of edf file
    edf_header = edf.getHeader()
    h5_headers = [
        "recording_additional",
        "patientname",
        "patientcode",
        "patient_additional",
        "gender",
        "equipment",
        "admincode",
        "birthdate",
        "technician",
    ]

    for h5_header in h5_headers:
        h5.attrs[h5_header] = edf_header.get(h5_header, "")

    # Add datetime and timezone informations
    edf_start_timestamp = edf_header.get("startdate").timestamp()
    edf_stop_timestamp = edf_start_timestamp + edf.getFileDuration()
    start_timestamp = edf_start_timestamp
    end_timestamp = edf_stop_timestamp

    if start_minute:
        start_timestamp = np.ceil(edf_start_timestamp / 60) * 60
        end_timestamp = np.floor(edf_stop_timestamp / 60) * 60
    if start_30s:
        start_timestamp = np.ceil(edf_start_timestamp / 30) * 30
        end_timestamp = np.floor(edf_stop_timestamp / 30) * 30
    if lights_off is not None:
        start_timestamp = max(edf_start_timestamp, lights_off)
    if lights_on is not None:
        end_timestamp = min(edf_stop_timestamp, lights_on)

    point_to_remove_at_start = int(100 * (start_timestamp - edf_start_timestamp))
    point_to_remove_at_end = int(100 * (edf_stop_timestamp - end_timestamp))

    h5.attrs["start_time"] = start_timestamp
    h5.attrs["duration"] = end_timestamp - start_timestamp
    h5.attrs["stop_time"] = end_timestamp

    derivations = build_derivation_setup_from_signal_headers(edf, electrodes)
    assert len(derivations) > 0, 'None of the provided derivations are in the edf file'

    is_compatible, missing_derivation = is_record_compatible_with_setup(derivations, signal_index)
    assert is_compatible, "The following channels are missing {} from the record".format(
        str(missing_derivation)
    )

    for i, derivation in enumerate(derivations):
        base = derivation["base"]
        signal_base = np.array(edf.readSignal(signal_index[base]))
        reference = derivation.get("reference", None)

        assert edf.getSampleFrequency(signal_index[base]) == edf.getSampleFrequency(
            signal_index.get(reference, signal_index[base])
        )

        if reference is not None:
            signal_reference = np.array(edf.readSignal(signal_index[reference]))
            signal_reference = poly_resample(signal_reference,
                                             source_fs=edf.getSampleFrequency(signal_index[reference]),
                                             target_frequency=edf.getSampleFrequency(signal_index[base]))

            signal = signal_base - signal_reference
            label = "{}-{}".format(derivation["base"], derivation["reference"])
        else:
            signal = signal_base
            label = clean_derivation(derivation["base"])

        path = "".join(label.replace(" ", "_").lower().split("-"))
        if path in list(h5):
            continue

        data = poly_resample(signal, source_fs=edf.getSampleFrequency(signal_index[base]),
                             target_frequency=100)
        if point_to_remove_at_start > 0:
            data = data[point_to_remove_at_start:]
        if point_to_remove_at_end > 0:
            data = data[:-point_to_remove_at_end]

        h5.create_dataset(path, data=data, dtype=np.float32, compression="gzip")

        for header, value in edf.getSignalHeader(signal_index[base]).items():
            h5[path].attrs[header] = value
        h5[path].attrs["location"] = label

        descriptor.append(
            {
                "name": label.replace(" ", "_"),
                "fs": 100,
                "path": "{}".format(path),
            }
        )

    h5.attrs.create("description", json.dumps(descriptor), dtype=np.dtype("S32768"))

    h5.close()
    return h5_filename, start_timestamp, end_timestamp, electrodes

