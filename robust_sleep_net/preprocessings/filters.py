from scipy.signal import iirfilter, filtfilt, lfilter


def iir_bandpass_filter(
    signal,
    fs=250.0,
    order=4,
    frequency_band=[0.4, 18],
    filter_type="butter",
    axis=-1,
    forward_backward=False,
):
    """ Perform bandpass filtering using scipy library.

    Parameters
    ----------

    signal : 1D numpy.array
        Array to filter.
    fs : float
        Sampling frequency
    order : int
        Order of the filter
    frequency_band : list
        Specify bandpass eg: [0.5, 20] will keep frequencies between 0.5
        and 20 Hz
    filter_type : str
        Choose type of IIR filter: butter, cheby1, cheby2, ellip, bessel
    axis: int
        Choose axis where to perform filtering.
    forward_backward : boolean
        Set True if you want a null phase shift filtered signal

    Returns
    -------

        1D numpy.array
            The signal filtered
    """
    b, a = iirfilter(
        order, [ff * 2.0 / fs for ff in frequency_band], btype="bandpass", ftype=filter_type,
    )
    if forward_backward:
        result = filtfilt(b, a, signal, axis)
    else:
        result = lfilter(b, a, signal, axis)
    return result
