from typing import List, Dict

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader

from models.mlp import get_mlp

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


def kfold_cv(k: int, X: np.ndarray, y: np.ndarray, num_neurons_options: List, num_epochs: int):
    kf = KFold(k, shuffle=True, random_state=RAND_SEED)

    batch_size = 32

    models_performance = {}

    for num_neurons in num_neurons_options:
        model = get_mlp(2, 1, num_neurons)
        model.train()
        mse = nn.MSELoss()
        optimizer = optim.SGD(model.parameters())

        fold_losses = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            X_train_tensor = torch.from_numpy(X_train)
            y_train_tensor = torch.from_numpy(y_train)
            X_val_tensor = torch.from_numpy(X_val)
            y_val_tensor = torch.from_numpy(y_val)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            for epoch in range(num_epochs):
                epoch_loss = 0.0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = mse(outputs, batch_y)
                    mse.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = mse(outputs, batch_y)
                    val_loss += loss.item()
            average_val_loss = val_loss / len(val_loader)
            fold_losses.append(average_val_loss)
            print(f'Validation Loss: {average_val_loss:.4f}')

        models_performance[num_neurons] = np.mean(fold_losses)


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


if __name__ == "__main__":
    main()
