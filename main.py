from typing import List, Tuple

import torch
import torch.nn as nn


import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader

from models.mlp import get_mlp
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
    abdomen1 = import_data_from_txt_to_np("./ECGdata/abdomen1.txt")[:2000]

    abdomen2 = import_data_from_txt_to_np("./ECGdata/abdomen2.txt")[:2000]
    abdomen3 = import_data_from_txt_to_np("./ECGdata/abdomen3.txt")[:2000]

    thorax1 = import_data_from_txt_to_np("./ECGdata/thorax1.txt")[:2000]
    thorax2 = import_data_from_txt_to_np("./ECGdata/thorax2.txt")[:2000]

    datasets = [abdomen1, abdomen2, abdomen3, thorax1, thorax2]

    # 1000 Hz taken from the assignment
    fs = 1000
    # The typical heart rate = around 60 to 100 bpm =~ 1 to 1.7 Hz. Anything below is noise.
    cutoff = 0.7

    # High-pass the data
    high_passed = [butter_low_high_pass_filter(data=dataset, cutoff=cutoff, fs=fs, order=2, high_low="high")
                   for dataset in datasets]

    # Normalize the data
    norm_datasets = [normalize(dataset) for dataset in high_passed]

    # # Plot the data
    # plot_data(datasets)
    # plot_data(high_passed)
    # plot_data(norm_datasets)

    # Data as tensors
    abdomen1_tensor = torch.from_numpy(norm_datasets[0]).unsqueeze(1)
    abdomen2_tensor = torch.from_numpy(norm_datasets[1]).unsqueeze(1)
    abdomen3_tensor = torch.from_numpy(norm_datasets[2]).unsqueeze(1)

    thorax1_tensor = torch.from_numpy(norm_datasets[3]).unsqueeze(1)
    thorax2_tensor = torch.from_numpy(norm_datasets[4]).unsqueeze(1)

    # Total data
    X = torch.cat((thorax1_tensor, thorax2_tensor), 1)
    Y = abdomen3_tensor

    # Training data
    index_90_percent = int(X.shape[0] * 0.9)
    X_train = X[:index_90_percent]
    Y_train = Y[:index_90_percent]

    # Test data
    X_test = X[index_90_percent:]
    Y_test = Y[index_90_percent:]

    print(thorax1_tensor.shape, thorax2_tensor.shape,
          X.shape, Y.shape,
          X_train.shape, Y_train.shape,
          X_test.shape, Y_test.shape)

    # Initialize the mlp
    num_hidden_layers = 1
    num_neurons = [10, 10, 10]

    loss = nn.MSELoss()

    model = get_mlp(num_input_first_layer=X.shape[1], num_hidden_layers=num_hidden_layers, num_neurons=num_neurons)

    # ------ LSTM model ------

    seq_length = 50
    X_train_seq, Y_train_seq = create_sequences(X_train.numpy(),
                                                Y_train.numpy(), seq_length)
    X_test_seq, Y_test_seq = create_sequences(X_test.numpy(),
                                              Y_test.numpy(), seq_length)

    X_train_tensor = torch.from_numpy(X_train_seq).float()
    Y_train_tensor = torch.from_numpy(Y_train_seq).float()
    X_test_tensor = torch.from_numpy(X_test_seq).float()
    Y_test_tensor = torch.from_numpy(Y_test_seq).float()

    train_data = TensorDataset(X_train_tensor, Y_train_tensor)
    test_data = TensorDataset(X_test_tensor, Y_test_tensor)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    # initiliaze the LSTM model
    input_size = X_train_tensor.shape[2]
    hidden_size = 128
    num_layers = 2
    output_size = 1
    droput_prob = 0.2

    lstm_trainer = LSTMTrainer(input_size, hidden_size,
                               num_layers, output_size,
                               dropout_prob=droput_prob)
    lstm_trainer.train(train_loader, test_loader, num_epochs=50, patience=5)

    model_evaluator = ModelEvaluator(lstm_trainer.get_model(), test_loader)
    model_evaluator.evaluate()
    model_evaluator.calculate_metrics()
    train_loss, val_loss = lstm_trainer.get_loss_history()

    # plot the evaluation
    model_evaluator.plot_predictions()
    model_evaluator.plot_loss(train_loss, val_loss)


if __name__ == "__main__":
    main()
