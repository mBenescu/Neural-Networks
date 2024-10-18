from typing import List, Tuple

import torch
import torch.nn as nn


import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from models.mlp import *
from models.lstm import LSTMTrainer, ModelEvaluator

RAND_SEED = 42
torch.manual_seed(RAND_SEED)


def import_data_from_txt_to_np(path: str) -> np.ndarray:
    """
    Loads the data as txt and stores it into an ndarray
    :param path: location of the data
    :return: data as ndarray
    """
    data_array = np.loadtxt(path, dtype=np.float64)
    return data_array


def normalize(dataset: np.ndarray) -> np.ndarray:
    """
    Normalize the data to 0 mean and unit variance
    :param dataset: the dataset to be normalized
    :return: the normalized dataset
    """

    dataset_norm = (dataset - dataset.mean()) / dataset.std()
    return dataset_norm


def plot_data(datasets: List[np.ndarray]) -> None:
    """
    Plots the datasets stacked vertically, sharing the x axis
    :param datasets:  the datasets to plot
    """
    fix, axs = plt.subplots(len(datasets), sharex=True)
    for i, dataset in enumerate(datasets):
        axs[i].plot(dataset)
    plt.show()


def butter_low_high_pass_filter(data, cutoff, fs, order, high_low="low"):
    """
    Applies a low or high pass filter to the data
    :param data: data to apply the filter on
    :param cutoff: the cutoff frequency (the frequency after/from which to allow frequencies to pass)
    :param fs: sampling frequency
    :param order: order of the filter
    :param high_low: "low" or "high", depending on what type of filtered is desired
    :return: the filtered data
    :Notes: For more info see:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype=high_low, analog=False)
    y = filtfilt(b, a, data)
    return y



def create_sequences(data: np.ndarray, labels: np.ndarray,
                     seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits time series data into sequences of a specified length along
    with their corresponding labels. This creates sliding window sequences
    of the input data and assigns the corresponding label to each sequence.
    :param data: the input time series data
    :param labels: the target values of the data
    :param seq_length: the length of each sequence
    :return xs: an array of input sequences (seq_length, n_features)
    :return ys: an array of labels
    """
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = labels[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def main():
    offset = 12

    abs_start_index, abs_end_index = 400, 2400
    thorax_start_index, thorax_end_index = abs_start_index - offset, abs_end_index - offset


    abdomen1 = import_data_from_txt_to_np("./ECGdata/abdomen1.txt")[abs_start_index:abs_end_index]

    abdomen2 = import_data_from_txt_to_np("./ECGdata/abdomen2.txt")[abs_start_index:abs_end_index]
    abdomen3 = import_data_from_txt_to_np("./ECGdata/abdomen3.txt")[abs_start_index:abs_end_index]

    thorax1 = import_data_from_txt_to_np("./ECGdata/thorax1.txt")[thorax_start_index:thorax_end_index]
    thorax2 = import_data_from_txt_to_np("./ECGdata/thorax2.txt")[thorax_start_index:thorax_end_index]


    datasets = [abdomen1, abdomen2, abdomen3, thorax1, thorax2]

    # 1000 Hz taken from the assignment
    fs = 1000
    # The typical heart rate = around 60 to 100 bpm =~ 1 to 1.7 Hz. Anything below is noise.
    cutoff = 0.5

    # High-pass the data
    high_passed = [butter_low_high_pass_filter(data=dataset, cutoff=cutoff, fs=fs, order=2, high_low="high")
                   for dataset in datasets]

    # Normalize the data
    norm_datasets = [normalize(dataset) for dataset in high_passed]

    # # Plot the data
    plot_data(datasets)
    plot_data(high_passed)
    plot_data(norm_datasets)

    abs3_norm = norm_datasets[2].reshape(-1, 1)
    thorax2_norm = norm_datasets[4].reshape(-1, 1)

    print(np.argmax(abs3_norm), np.argmax(thorax2_norm))

    linear_regression = LinearRegression()

    linear_regression.fit(thorax2_norm, abs3_norm)

    filtered_output = abs3_norm - linear_regression.predict(thorax2_norm).reshape(abs3_norm.shape)

    plot_data([filtered_output, abs3_norm - thorax2_norm])








if __name__ == "__main__":
    main()
