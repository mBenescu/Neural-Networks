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


def main():
    offset = 12

    abs_start_index, abs_end_index = 400, 2400
    thorax_start_index, thorax_end_index = abs_start_index , abs_end_index #- offset

    abdomen1 = import_data_from_txt_to_np("./ECGdata/abdomen1.txt")[abs_start_index:abs_end_index]

    abdomen2 = import_data_from_txt_to_np("./ECGdata/abdomen2.txt")[abs_start_index:abs_end_index]
    abdomen3 = import_data_from_txt_to_np("./ECGdata/abdomen3.txt")[abs_start_index:abs_end_index]

    thorax1 = import_data_from_txt_to_np("./ECGdata/thorax1.txt")[thorax_start_index:thorax_end_index]
    thorax2 = import_data_from_txt_to_np("./ECGdata/thorax2.txt")[thorax_start_index:thorax_end_index]

    datasets = [abdomen1, abdomen2, abdomen3, thorax1, thorax2]

    # 1000 Hz taken from the assignment
    fs = 1000
    # The typical heart rate = around 60 to 100 bpm =~ 1 to 1.7 Hz. Anything below is noise.
    cutoff = .5

    # High-pass the data
    high_passed = [butter_low_high_pass_filter(data=dataset, cutoff=cutoff, fs=fs, order=2, high_low="high")
                   for dataset in datasets]

    # Plot the frequency domain of the signals
    # plot_fft(abdomen3, 1000, "Abs3 raw")
    # plot_fft(high_passed[2], 1000, "Abs3 high-passed")
    # plot_fft(thorax2, 1000, "Thorax2 raw")
    # plot_fft(high_passed[-1], 1000, "Thorax2 high-passed")

    # Normalize the data

    # Subtract mean to remove DC offset
    datasets_zero_mean = [dataset - np.mean(dataset) for dataset in high_passed]

    # Scale signals uniformly
    max_abs_value = max([np.max(np.abs(dataset)) for dataset in datasets_zero_mean])
    datasets_scaled = [dataset / max_abs_value for dataset in datasets_zero_mean]

    # Prepare signals for filtering
    abs3 = datasets_scaled[2]
    thorax2 = datasets_scaled[4]

    # abs3_norm = normalize(high_passed[2], mean=mean, std=std).reshape(-1, 1)
    # thorax2_norm = normalize(high_passed[-1], mean=mean, std=std).reshape(-1, 1)
    thorax2_high_passed = high_passed[-1].reshape(-1, 1)
    abs3_high_passed = high_passed[2].reshape(-1, 1)

    # norm_datasets = [abs3_norm, thorax2_norm]

    # # # Plot the data
    plot_data(datasets)
    plot_data(high_passed)
    # plot_data(norm_datasets)


    # print(np.argmax(abs3_norm), np.argmax(thorax2_norm))

    linear_regression = LinearRegression()

    # linear_regression.fit(thorax2_norm, abs3_norm)
    linear_regression.fit(thorax2_high_passed, abs3_high_passed)

    # filtered_output_regression = abs3_norm - linear_regression.predict(thorax2_norm).reshape(abs3_norm.shape)
    filtered_output_regression = abs3_high_passed - linear_regression.predict(thorax2_high_passed).\
        reshape(abs3_high_passed.shape)

    filter_length = 50
    learning_rate = 0.001

    initial_weights = np.zeros(filter_length)
    fir = FIR_filter(initial_weights)
    # y = np.zeros(len(abs3_norm), dtype=np.float64)
    # y = np.zeros(len(abs3_high_passed), dtype=np.float64)
    y = np.zeros(len(abs3), dtype=np.float64)

    # abs3_norm = abs3_norm.flatten()
    # abs3_high_passed = abs3_high_passed.flatten()
    abs3 = abs3.flatten()
    # thorax2_norm = thorax2_norm.flatten()
    # thorax2_high_passed = thorax2_high_passed.flatten()
    thorax2 = thorax2.flatten()

    # for i in range(len(abs3_norm)):
    # for i in range(len(abs3_high_passed)):
    for i in range(len(abs3)):
        # canceller = fir.filter(thorax2_norm[i])
        # canceller = fir.filter(thorax2_high_passed[i])
        canceller = fir.filter(thorax2[i])
        # output_signal = abs3_norm[i] - canceller
        # output_signal = abs3_high_passed[i] - canceller
        output_signal = abs3[i] - canceller
        if i % 100 == 0:
            # print(f"Output Signal: {output_signal}, Canceller: {canceller}, Input: {thorax2_norm[i]}")
            # print(f"Output Signal: {output_signal}, Canceller: {canceller}, Input: {thorax2_high_passed[i]}")
            print(f"Output Signal: {output_signal}, Canceller: {canceller}, Input: {thorax2[i]}")
        fir.lms(output_signal, learning_rate)
        y[i] = output_signal

    plot_data([filtered_output_regression, y])


if __name__ == "__main__":
    main()
