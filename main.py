from numbers import Number
from typing import Tuple, Any, Dict

import numpy as np
from scipy.signal import butter, filtfilt, freqz
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import seaborn as sns

from models.mlp import *
from models.filters import FIR_filter

RAND_SEED = 42


def create_X_Y(abs_data: np.ndarray, thorax_data: np.ndarray, filter_window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs the input matrix X and target vector Y for the linear regression model using a sliding window approach.

    :param abs_data: The abdominal ECG data (abdomen3 channel) as a numpy array.
    :param thorax_data: The thoracic ECG data (thorax2 channel) as a numpy array.
    :param filter_window: The length of the filter window (number of data points to consider).
    :return: A tuple (X, Y) where:
             - X: Input matrix for the regression model, where each row is a window of thorax_data points plus a bias term.
             - Y: Target vector, consisting of the corresponding points from abs_data.
    """
    print(f"{filter_window=}")
    index = filter_window // 2
    X = np.ones((len(abs_data) - filter_window, filter_window + 1))
    Y = abs_data[index: -index]
    for i in range(len(Y)):
        X[i, : -1] = thorax_data[i: filter_window + i].flatten()
    return X, Y


def plot_fft(signal: np.ndarray, fs: int, title: str = "the given signal") -> None:
    """
    Plots the FFT of a given signal.
    :param signal: Input signal to be analyzed in the frequency domain.
    :param fs: Sampling frequency (in Hz).
    :param title: Title for the plot.
    """
    # Compute the FFT
    fft_result = np.fft.fft(signal)
    # Compute the corresponding frequencies
    freqs = np.fft.fftfreq(len(signal), d=1 / fs)
    # Only take the positive part of the spectrum
    positive_freqs = freqs[:len(freqs) // 2]
    magnitude = np.abs(fft_result)[:len(freqs) // 2]

    # Plot the magnitude spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(positive_freqs, magnitude)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f'Frequency Spectrum of {title}')
    plt.grid(True)
    plt.show()


def import_data_from_txt_to_np(path: str) -> np.ndarray:
    """
    Loads the data as txt and stores it into an ndarray
    :param path: location of the data
    :return: data as ndarray
    """
    data_array = np.loadtxt(path, dtype=np.float64)
    return data_array


def normalize(dataset: np.ndarray, mean: np.float64, std: np.float64) -> np.ndarray:
    """
    Normalize the data to 0 mean and unit variance
    :param std: standard deviation to divide the dataset
    :param mean: mean to subtract
    :param dataset: the dataset to be normalized
    :return: the normalized dataset
    """

    dataset_norm = (dataset - mean) / std
    return dataset_norm


def plot_data(datasets: List[np.ndarray | List[Number]], titles: List[str], general_title: str,
              x_axis: List[Any] = None) -> None:
    """
    Plots the datasets stacked vertically, sharing the x axis
    :param x_axis: The x axis of the plot
    :param datasets: the datasets to plot
    :param titles: the titles of each subplot
    :param general_title: the general title for the entire figure
    Plots the datasets stacked vertically, sharing the x axis
    :param datasets: the datasets to plot
    """
    fig, axs = plt.subplots(len(datasets), sharex=True, figsize=(10, 6))
    fig.suptitle(general_title, fontsize=16)

    for i, (dataset, title) in enumerate(zip(datasets, titles)):
        axs[i].plot(x_axis, dataset) if x_axis is not None else axs[i].plot(dataset)
        axs[i].text(1.05, 0.5, title, va='center', ha='left', rotation='horizontal',
                    transform=axs[i].transAxes, fontsize=10)
        axs[i].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_final_ecg(normalized_y_test: np.ndarray, normalized_baby_ecg: np.ndarray, title: str) -> None:
    """
    Plots the normalized abs3 channel and normalized baby ECG on the same plot.

    :param normalized_y_test: Normalized abs3 channel data (numpy array).
    :param normalized_baby_ecg: Normalized baby ECG data (numpy array).
    :param title: Title of the plot.
    """
    plt.figure(figsize=(15, 6))

    # Plot normalized abs3 channel in red
    plt.plot(normalized_y_test, color='red', label='Normalized Abdomen channel (Maternal + Fetal ECG)', alpha=0.7)

    # Plot normalized baby ECG in black
    plt.plot(normalized_baby_ecg, color='black', label='Normalized Smoothed Residuals (Fetal ECG)', alpha=0.7)

    plt.xlabel('Sample Index')
    plt.ylabel('Normalized Amplitude')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
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


def get_heatmap_matrix(results_by_filter_length: Dict, key: str):
    """
    Constructs a matrix suitable for plotting a heatmap from the results dictionary.

    :param results_by_filter_length: A dictionary where each key is a filter length, and the value is a dictionary
                                     containing performance metrics and corresponding alpha values.
    :param key: The metric to extract from the results dictionary (e.g., 'mse_train', 'mse_test').
    :return: A 2D numpy array containing the specified metric, where rows correspond to filter lengths and
             columns to alpha values.
    """
    # Prepare data for heatmap
    alphas = sorted(set(alpha for data in results_by_filter_length.values() for alpha in data['alphas']))
    filter_lengths = sorted(results_by_filter_length.keys())
    heatmap_matrix = np.zeros((len(filter_lengths), len(alphas)))

    for i, filter_length in enumerate(filter_lengths):
        data = results_by_filter_length[filter_length]
        info_to_plot = data[key]
        alpha_indices = [alphas.index(alpha) for alpha in data['alphas']]
        heatmap_matrix[i, alpha_indices] = info_to_plot

    return heatmap_matrix


def plot_heatmap(results_by_filter_length: Dict, key: str, title: str) -> None:
    """
    Plots a heatmap for a specified metric over different filter lengths and alpha values.

    :param results_by_filter_length: A dictionary containing results organized by filter lengths. Each filter length
                                     maps to a dictionary with metrics and alpha values.
    :param key: The metric to plot in the heatmap (e.g., 'mse_train', 'mse_test').
    :param title: The title of the heatmap plot.
    """

    matrix_to_plot = get_heatmap_matrix(results_by_filter_length, key)
    alphas = sorted(set(alpha for data in results_by_filter_length.values() for alpha in data['alphas']))
    filter_lengths = sorted(results_by_filter_length.keys())
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix_to_plot, annot=True, fmt=".4f",
                xticklabels=alphas, yticklabels=filter_lengths, cmap="viridis")
    plt.xlabel("Alpha")
    plt.ylabel("Filter Length")
    plt.title(title)
    plt.show()


def plot_comparison_heatmaps(results_by_filter_length: Dict, keys: Tuple[str, str], titles: Tuple[str, str]) -> None:
    """
    Plots two heatmaps side by side for training and test metrics over different filter lengths and alpha values.

    :param results_by_filter_length: A dictionary containing results organized by filter lengths. Each filter length
                                     maps to a dictionary with metrics and alpha values.
    :param keys: A tuple containing the keys for the metrics to plot (e.g., ('mse_train', 'mse_test')).
    :param titles: A tuple containing titles for the two heatmaps (e.g., ("MSE Train", "MSE Test")).
    """
    alphas = sorted(set(alpha for data in results_by_filter_length.values() for alpha in data['alphas']))
    filter_lengths = sorted(results_by_filter_length.keys())

    metric_train, metric_test = keys
    title_train, title_test = titles

    # Get the matrices for training and test MSE
    mse_train_matrix = get_heatmap_matrix(results_by_filter_length, metric_train)
    mse_test_matrix = get_heatmap_matrix(results_by_filter_length, metric_test)

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Plot Training MSE Heatmap
    sns.heatmap(mse_train_matrix, annot=True, fmt=".4f", ax=axes[0],
                xticklabels=alphas, yticklabels=filter_lengths, cmap="viridis")
    axes[0].set_xlabel("Alpha")
    axes[0].set_ylabel("Filter Length")
    axes[0].set_title(title_train)

    # Plot Test MSE Heatmap
    sns.heatmap(mse_test_matrix, annot=True, fmt=".4f", ax=axes[1],
                xticklabels=alphas, yticklabels=filter_lengths, cmap="viridis")
    axes[1].set_xlabel("Alpha")
    # axes[1].set_ylabel("Filter Length")
    axes[1].set_title(title_test)

    plt.tight_layout()
    plt.show()


def main():
    abdomen1 = import_data_from_txt_to_np("./ECGdata/abdomen1.txt")

    abdomen2 = import_data_from_txt_to_np("./ECGdata/abdomen2.txt")
    abdomen3 = import_data_from_txt_to_np("./ECGdata/abdomen3.txt")

    thorax1 = import_data_from_txt_to_np("./ECGdata/thorax1.txt")
    thorax2 = import_data_from_txt_to_np("./ECGdata/thorax2.txt")

    datasets = [abdomen1, abdomen2, abdomen3, thorax1, thorax2]

    # 1000 Hz taken from the assignment
    fs = 1000
    # The typical heart rate = around 60 to 100 bpm =~ 1 to 1.7 Hz. Anything below is noise.
    high_cutoff = 0.5

    # High-pass the data
    high_passed = [butter_low_high_pass_filter(data=dataset, cutoff=high_cutoff, fs=fs, order=2, high_low="high")
                   for dataset in datasets]

    # Plot the frequency domain of the signals
    plot_fft(abdomen3, 1000, "Abdomen3 channel raw")
    plot_fft(high_passed[2], 1000, "Abdomen3 channel high-passed")
    # plot_fft(low_passed[2], 1000, "Abs3 low-passed")
    plot_fft(thorax2, 1000, "Thorax2 channel raw")
    plot_fft(high_passed[-1], 1000, "Thorax2 channel high-passed")
    # plot_fft(low_passed[-1], 1000, "Thorax2 low-passed")

    # Prepare signals for filtering
    abs3 = high_passed[2].reshape(-1, 1)
    thorax2 = high_passed[4].reshape(-1, 1)

    # # # Plot the data
    titles = ["abdomen1", "abdomen2", "abdomen3", "thorax1", "thorax2"]
    plot_data(datasets, titles, "Raw Data")
    plot_data(high_passed, titles, "High-passed with a cutoff frequency of " + str(high_cutoff) + " Hz")
    # plot_data(low_passed, titles, "Low-passed with a cutoff frequency of " + str(low_cutoff) + " Hz")
    # plot_data(norm_datasets, titles, "Individually normalized dataset")
    # print(np.argmax(abs3_norm), np.argmax(thorax2_norm))

    # alphas = [0, 0.01, 0.1, 1, 10, 100]
    # # filter_lengths = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000]
    #
    # filter_lengths = [i for i in range(430, 461, 2)]
    #
    # # Initialize storage for plotting results
    # results_by_filter_length = {length: {'alphas': [], 'mse_train': [], 'mse_test': [], 'r2_score_train': [],
    #                                      'r2_score_test': []} for length in filter_lengths}
    # best_alpha, best_filter_length, lowest_mse = None, None, float('inf')
    #
    # for filter_length in filter_lengths:
    #     total_X, total_y = create_X_Y(abs3, thorax2, filter_length)
    #     split_index = int(len(total_X) * 0.9)
    #
    #     # Split the data
    #     X_train, y_train = total_X[: split_index], total_y[:split_index]
    #     X_test, y_test = total_X[split_index:], total_y[split_index:]
    #
    #     for alpha in alphas:
    #         print(f"{alpha =}")
    #         # Train Ridge Regression
    #         model = Ridge(alpha=alpha)
    #         model.fit(X_train, y_train)
    #
    #         # Predictions and evaluation
    #         prediction_train = model.predict(X_train)
    #         prediction_test = model.predict(X_test)
    #         mse_train = mean_squared_error(y_train, prediction_train)
    #         mse_test = mean_squared_error(y_test, prediction_test)
    #         r2_score_train = model.score(X_train, y_train)
    #         r2_score_test = model.score(X_test, y_test)
    #
    #         # Track the best parameters based on MSE on the test set
    #         if mse_test < lowest_mse:
    #             best_alpha, best_filter_length, lowest_mse = alpha, filter_length, mse_test
    #
    #         # Store results for this filter length
    #         results_by_filter_length[filter_length]['alphas'].append(alpha)
    #         results_by_filter_length[filter_length]['mse_train'].append(mse_train)
    #         results_by_filter_length[filter_length]['mse_test'].append(mse_test)
    #         results_by_filter_length[filter_length]['r2_score_train'].append(r2_score_train)
    #         results_by_filter_length[filter_length]['r2_score_test'].append(r2_score_test)
    #
    # # Plotting results
    # plot_mse_heatmaps(results_by_filter_length, keys=("mse_train", "mse_test"), titles=("MSE Train", "MSE Test"))
    # plot_mse_heatmaps(results_by_filter_length, keys=("r2_score_train", "r2_score_test"), titles=("R^2 Score Train",
    #                                                                                               "R^2 Score Test"))
    #
    # print(f"Best alpha: {best_alpha}, Best filter length: {best_filter_length}, Lowest MSE test: {lowest_mse}")

    # Train the best model
    best_filter_length = 444
    total_X, total_y = create_X_Y(abs3, thorax2, best_filter_length)
    # split_index = int(len(total_X) * 0.1)

    # Split the data
    # X_train, y_train = total_X[split_index:], total_y[split_index:]
    # X_test, y_test = total_X[:split_index], total_y[:split_index]

    best_model = Ridge(alpha=0)
    best_model.fit(total_X, total_y)

    test_prediction = best_model.predict(total_X)
    residuals = total_y - test_prediction

    squared_residuals = residuals ** 2

    data_to_plot = [test_prediction, residuals, squared_residuals]

    plot_data(data_to_plot, ["Model Prediction", "Signal Residuals (Baby's ECG)",
                             "Squared Residuals"], "")

    plot_fft(squared_residuals, fs, "Squared Residuals in Frequency Domain")

    smoothed_residuals = butter_low_high_pass_filter(data=squared_residuals.flatten(), cutoff=30, fs=fs, order=2,
                                                     high_low="low")

    plot_fft(smoothed_residuals, fs, "Smoothed Squared Residuals in Frequency Domain")

    # normalized_y_test = normalize(y_test, y_test.mean(), y_test.std())
    normalized_y = normalize(abdomen3, abdomen3.mean(), abdomen3.std())

    plt.show()

    normalized_smoothed_residuals = normalize(smoothed_residuals, smoothed_residuals.mean(), smoothed_residuals.std())

    plot_data([smoothed_residuals, normalized_smoothed_residuals], ["Smoothed Squared Residuals (30Hz) ",
                                                                    "Normalized signal"], "Final Results")

    plot_final_ecg(
        normalized_y,
        normalized_smoothed_residuals,
        "Abdomen Channel and Baby's ECG Normalized to Zero Mean and Unit Variance"
    )


if __name__ == "__main__":
    main()
