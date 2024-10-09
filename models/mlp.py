from typing import List
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
import random

RAND_SEED = 42
torch.manual_seed(RAND_SEED)
random.seed(RAND_SEED)


def get_mlp(num_input_first_layer: int, num_neurons: List[int]) -> torch.nn.Sequential:
    """
    Creates an MLP with the given parameters
    :param num_input_first_layer: number of input data, number of features
    :param num_neurons: a list containing the number of neurons of each non-output layer
    :return: the MLP
    """

    num_hidden_layers = len(num_neurons)

    model = torch.nn.Sequential()

    model.add_module("linear_0", nn.Linear(num_input_first_layer, num_neurons[0]))

    for i in range(1, num_hidden_layers):
        model.add_module(f"linear_{i}", nn.Linear(num_neurons[i - 1], num_neurons[i]))
        model.add_module(f"tanh_{i}", nn.Tanh())

    model.add_module("linear_last", nn.Linear(num_neurons[-1], 1))

    return model


def train_epoch(model: nn.Sequential, optimizer: optim.Optimizer, criterion: nn.Module
                , train_loader: DataLoader) -> float:
    """
    Trains the model one epoch
    :param model: the model to train
    :param optimizer: optimizer of the training procedure
    :param criterion: loss function
    :param train_loader: data loader
    :return: the loss of the model on that epoch
    """
    model.train()
    running_train_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * batch_X.size(0)  # Multiply by batch size to get a weighted sum

    # Calculate average training loss for the epoch
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    return epoch_train_loss


def eval_epoch(model: nn.Sequential, criterion: nn.Module, val_loader: DataLoader) -> float:
    """
    Evaluates the model on one epoch
    :param model: model to evaluate
    :param criterion: loss function
    :param val_loader: validation data loader
    :return: the validation loss on that epoch
    """
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            running_val_loss += loss.item() * batch_X.size(0)

    # Calculate average validation loss for the epoch
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    return epoch_val_loss


def kfold_cv(k: int, X: np.ndarray, y: np.ndarray, num_neurons_options: List[List[int]], num_epochs: int):
    kf = KFold(k, shuffle=True, random_state=RAND_SEED)
    batch_size = 32
    models_performance = {}

    for num_neurons in num_neurons_options:
        print(f"\nEvaluating model with neurons: {num_neurons}")
        fold_losses = []
        fold_train_losses = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\nFold {fold + 1}/{k}")

            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Convert to tensors
            X_train_tensor = torch.from_numpy(X_train).float()
            y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1)
            X_val_tensor = torch.from_numpy(X_val).float()
            y_val_tensor = torch.from_numpy(y_val).float().unsqueeze(1)

            # Create datasets and loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            # Initialize model and optimizer within the fold
            model = get_mlp(num_input_first_layer=X.shape[1], num_neurons=[num_neurons])
            criterion = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)

            # Lists to store epoch-wise losses
            epoch_train_losses = []
            epoch_val_losses = []

            for epoch in range(num_epochs):
                epoch_train_loss = train_epoch(model, optimizer, criterion, train_loader)
                epoch_train_losses.append(epoch_train_loss)

                # Validation after each epoch
                epoch_val_loss = eval_epoch(model, criterion, val_loader)
                epoch_val_losses.append(epoch_val_loss)

                print(f"Epoch {epoch + 1}/{num_epochs},"
                      f" Training Loss: {epoch_train_loss:.4f},"
                      f" Validation Loss: {epoch_val_loss:.4f}")

            # Store the final validation loss for this fold
            fold_losses.append(epoch_val_losses[-1])
            fold_train_losses.append(epoch_train_losses[-1])

        # Calculate average performance across folds
        mean_val_loss = np.mean(fold_losses)
        mean_train_loss = np.mean(fold_train_losses)

        print(f"\nAverage Validation Loss for neurons {num_neurons}: {mean_val_loss:.4f}")
        models_performance[num_neurons] = {
            'val_loss_mean': mean_val_loss,
            'train_loss_mean': mean_train_loss
        }

    return models_performance
