import numpy as np
from scipy.signal import butter, filtfilt, freqz
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from models.mlp import *
from models.filters import FIR_filter

RAND_SEED = 42
torch.manual_seed(RAND_SEED)


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


def plot_data(datasets: List[np.ndarray], titles: List[str], general_title: str) -> None:
    """
    Plots the datasets stacked vertically, sharing the x axis
    :param datasets: the datasets to plot
    :param titles: the titles of each subplot
    :param general_title: the general title for the entire figure
    Plots the datasets stacked vertically, sharing the x axis
    :param datasets: the datasets to plot
    """
    fig, axs = plt.subplots(len(datasets), sharex=True, figsize=(10, 6))
    fig.suptitle(general_title, fontsize=16)

    for i, (dataset, title) in enumerate(zip(datasets, titles)):
        axs[i].plot(dataset)
        axs[i].text(1.05, 0.5, title, va='center', ha='left', rotation='horizontal',
                    transform=axs[i].transAxes, fontsize=10)
        axs[i].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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


def main():
    offset = 12

    abs_start_index, abs_end_index = 0, 10000
    thorax_start_index, thorax_end_index = abs_start_index, abs_end_index #- offset

    abdomen1 = import_data_from_txt_to_np("./ECGdata/abdomen1.txt")[abs_start_index:abs_end_index]

    abdomen2 = import_data_from_txt_to_np("./ECGdata/abdomen2.txt")[abs_start_index:abs_end_index]
    abdomen3 = import_data_from_txt_to_np("./ECGdata/abdomen3.txt")[abs_start_index:abs_end_index]

    thorax1 = import_data_from_txt_to_np("./ECGdata/thorax1.txt")[thorax_start_index:thorax_end_index]
    thorax2 = import_data_from_txt_to_np("./ECGdata/thorax2.txt")[thorax_start_index:thorax_end_index]

    datasets = [abdomen1, abdomen2, abdomen3, thorax1, thorax2]

    # 1000 Hz taken from the assignment
    fs = 1000
    # The typical heart rate = around 60 to 100 bpm =~ 1 to 1.7 Hz. Anything below is noise.
    high_cutoff = 0.5

    low_cutoff = 40

    # High-pass the data
    high_passed = [butter_low_high_pass_filter(data=dataset, cutoff=high_cutoff, fs=fs, order=2, high_low="high")
                   for dataset in datasets]

    # Low-pass the data
    low_passed = [butter_low_high_pass_filter(data=dataset, cutoff=low_cutoff, fs=fs, order=2, high_low="low")
                  for dataset in high_passed]

    # Plot the frequency domain of the signals
    # plot_fft(abdomen3, 1000, "Abs3 raw")
    # plot_fft(high_passed[2], 1000, "Abs3 high-passed")
    # plot_fft(low_passed[2], 1000, "Abs3 low-passed")
    # plot_fft(thorax2, 1000, "Thorax2 raw")
    # plot_fft(high_passed[-1], 1000, "Thorax2 high-passed")
    # plot_fft(low_passed[-1], 1000, "Thorax2 low-passed")

    # Normalize the data

    all_data = np.array(datasets).flatten()
    global_mean = all_data.mean()
    global_std = all_data.std()

    norm_datasets = [normalize(dataset, dataset.mean(), dataset.std()) for dataset in low_passed]


    # Prepare signals for filtering
    abs3 = norm_datasets[2].reshape(-1, 1)
    thorax2 = norm_datasets[4].reshape(-1, 1)


    # # # Plot the data
    titles = ["abdomen1", "abdomen2", "abdomen3", "thorax1", "thorax2"]
    plot_data(datasets, titles, "Raw Data")
    plot_data(high_passed, titles, "High-passed with a cutoff frequency of " + str(high_cutoff) + " Hz")
    plot_data(low_passed, titles, "Low-passed with a cutoff frequency of " + str(low_cutoff) + " Hz")
    plot_data(norm_datasets, titles, "Individually normalized dataset")
    # print(np.argmax(abs3_norm), np.argmax(thorax2_norm))

    linear_regression = LinearRegression()

    linear_regression.fit(abs3, thorax2)

    prediction = linear_regression.predict(abs3)

    filtered_output_regression = abs3 - prediction.\
        reshape(abs3.shape)

    filter_length = 100

    learning_rate = 0.001

    initial_weights = np.zeros(filter_length)
    fir = FIR_filter(initial_weights)

    y = np.zeros(len(abs3), dtype=np.float64)

    abs3 = abs3.flatten()
    thorax2 = thorax2.flatten()

    for i in range(len(abs3)):
        canceller = fir.filter(thorax2[i])
        output_signal = abs3[i] - canceller
        if i % 100 == 0:
            print(f"Output Signal: {output_signal}, Canceller: {canceller}, Input: {thorax2[i]}")
        fir.lms(output_signal, learning_rate)
        y[i] = output_signal

    titles = ["abs3- LR prediction", "LR prediction", "norm thorax2", "norm abs3", "FIR length=" +
              str(filter_length) + " lr=" + str(learning_rate)]
    plot_data([filtered_output_regression, prediction, thorax2, abs3, y], titles, "Final results")


if __name__ == "__main__":
    main()
