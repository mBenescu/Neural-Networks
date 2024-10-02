from typing import List

import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


def import_data_from_txt_to_np(path: str) -> np.ndarray:
    """
    Loads the data as txt and stores it into an ndarray
    :param path: location of the data
    :return: data as ndarray
    """
    data_array = np.loadtxt(path, dtype=np.float64)
    return data_array


def plot_data(datasets: List[np.ndarray]) -> None:
    """
    Plots the datasets stacked vertically, sharing the x axis
    :param datasets:  the datasets to plot
    """
    fix, axs = plt.subplots(len(datasets), sharex=True)
    for i, dataset in enumerate(datasets):
        axs[i].plot(dataset)
    plt.show()


def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def main():
    abdomen1 = import_data_from_txt_to_np("./ECGdata/abdomen1.txt")

    abdomen2 = import_data_from_txt_to_np("./ECGdata/abdomen2.txt")
    abdomen3 = import_data_from_txt_to_np("./ECGdata/abdomen3.txt")

    thorax1 = import_data_from_txt_to_np("./ECGdata/thorax1.txt")
    thorax2 = import_data_from_txt_to_np("./ECGdata/thorax2.txt")

    datasets = [abdomen1, abdomen2, abdomen3, thorax1, thorax2]

    plot_data(datasets)






    # # Filter requirements.
    # T = 5.0  # Sample Period
    # fs = 30.0  # sample rate, Hz
    # cutoff = 2  # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
    # nyq = 0.5 * fs  # Nyquist Frequency
    # order = 2  # sin wave can be approx represented as quadratic
    # n = int(T * fs)  # total number of samples
    #
    # # sin wave
    # sig = np.sin(1.2 * 2 * np.pi * t)
    # # Lets add some noise
    # noise = 1.5 * np.cos(9 * 2 * np.pi * t) + 0.5 * np.sin(12.0 * 2 * np.pi * t)
    # data = sig + noise


# def plot():
#     # Filter the data, and plot both the original and filtered signals.
#     y = butter_lowpass_filter(data, cutoff, fs, order)
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         y=data,
#         line=dict(shape='spline'),
#         name='signal with noise'
#     ))
#     fig.add_trace(go.Scatter(
#         y=y,
#         line=dict(shape='spline'),
#         name='filtered signal'
#     ))
#     fig.show()


if __name__ == "__main__":
    main()
