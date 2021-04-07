import numpy as np
from scipy.signal import resample_poly

from .filters import iir_bandpass_filter


def poly_resample(signal, signal_properties, target_frequency):
    signal_frequency = signal_properties["fs"]
    if signal_frequency != target_frequency:
        signal_duration = signal.shape[0] / signal_frequency
        resampled_length = round(signal_duration * target_frequency)
        resampled_signal = resample_poly(signal, target_frequency, signal_frequency, axis=0)
        if len(resampled_signal) < resampled_length:
            padding = np.zeros((resampled_length - len(resampled_signal), signal.shape[-1]))
            resampled_signal = np.concatenate([resampled_signal, padding])
        signal_properties["fs"] = target_frequency
        return resampled_signal, signal_properties
    else:
        return signal, signal_properties


def filter_bandpass(signal, signal_properties, band=[0.2, 30], order=2):
    signal = iir_bandpass_filter(
        signal,
        axis=0,
        fs=signal_properties["fs"],
        order=order,
        forward_backward=True,
        frequency_band=band,
    )
    return signal, signal_properties


def pad_signal(signal, signal_properties, padding_duration, value=0):
    if padding_duration == 0:
        return signal, signal_properties
    else:
        if isinstance(value, int):
            fs = signal_properties["fs"]
            padding_array = np.zeros((padding_duration * fs,) + signal.shape[1:]) + value
            signal = [padding_array] + [signal] + [padding_array]
            signal_properties = {
                "fs": fs,
                "padding": signal_properties["padding"] + padding_duration,
            }
            return np.concatenate(signal), signal_properties
        if value == "min":
            fs = signal_properties["fs"]
            padding_array = np.zeros((padding_duration * fs,) + signal.shape[1:]) + np.min(signal)
            signal = [padding_array] + [signal] + [padding_array]
            signal_properties = {
                "fs": fs,
                "padding": signal_properties["padding"] + padding_duration,
            }
            return np.concatenate(signal), signal_properties


def weighted_sum(signal, signal_properties, weights=None):
    if weights is None:
        return np.sum(signal, -1, keepdims=True), signal_properties
    else:
        return np.sum(signal * np.array(weights), -1, keepdims=True), signal_properties


def rescale(signal, signal_properties, scale=1e3, bias=0):
    return signal * scale + bias, signal_properties


def normalize_signal_IQR(signal, signal_properties, clip_value=None, clip_IQR=None, eps=1e-5):
    if clip_value is not None:
        clipped_signal = np.clip(signal, a_min=-clip_value, a_max=clip_value)
    elif clip_IQR is not None:
        s_low = np.percentile(signal, 50, axis=0, keepdims=True) - np.percentile(
            signal, 25, axis=0, keepdims=True
        )
        s_high = np.percentile(signal, 75, axis=0, keepdims=True) - np.percentile(
            signal, 50, axis=0, keepdims=True
        )
        clipped_signal = np.clip(signal, a_min=-2 * clip_IQR * s_low, a_max=2 * clip_IQR * s_high)

    else:
        clipped_signal = signal

    mu = np.median(clipped_signal, axis=0, keepdims=True)
    sigma = np.percentile(clipped_signal, 75, axis=0, keepdims=True) - np.percentile(
        clipped_signal, 25, axis=0, keepdims=True
    )
    sigma[sigma == 0] = eps
    signal = (clipped_signal - mu) / (sigma)
    return signal, signal_properties


signal_processings = {
    "resample": poly_resample,
    "poly_resample": poly_resample,
    "filter_bandpass": filter_bandpass,
    "padding": pad_signal,
    "weighted_sum": weighted_sum,
    "rescale": rescale,
    "normalize_signal_IQR": normalize_signal_IQR,
}
