"""
File: lstm.py
Description: The file trains and evaluates an LSTM model.
Sources: For designing the model in this file the following
website was used to understand the workflow of Pytorch LSTM:
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
"""
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score


root = Path(__file__).resolve().parent.parent

# set up logs for the model
logs_dir = root / 'logs'
logs_dir.mkdir(parents=True, exist_ok=True)
logs_path = logs_dir / 'lstm.log'

format_style = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename=logs_path, level=logging.INFO,
                    format=format_style)


class Logger:
    """Simple logger file to store the information."""
    @staticmethod
    def log_info(message):
        logging.info(message)

    @staticmethod
    def log_error(message):
        logging.error(message)


class LSTMTrainer:
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int, output_size: int,
                 learning_rate: float = 0.0001,
                 dropout_prob: float = 0.2) -> None:
        self.model = LSTMModel(input_size, hidden_size,
                               num_layers, output_size, dropout_prob)
        self.criterion = nn.MSELoss()
        self.optimzer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.train_loss_history = []
        self.val_loss_history = []
        Logger.log_info("LSTM training is initialized.")

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 50, patience: int = 5) -> None:
        """Function to train the model."""
        best_val_loss = float('inf')
        epochs_improvement = 0

        try:
            self.model.train()
            for epoch in range(num_epochs):
                epoch_train_loss = 0.0
                epoch_val_loss = 0.0

                for batch_X, batch_y in train_loader:
                    self.optimzer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimzer.step()
                    epoch_train_loss += loss.item()

                avg_train_loss = epoch_train_loss / len(train_loader)
                self.train_loss_history.append(avg_train_loss)

                self.model.eval()
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        val_outputs = self.model(batch_X)
                        val_loss = self.criterion(val_outputs, batch_y)
                        epoch_val_loss += val_loss.item()

                    avg_val_loss = epoch_val_loss / len(val_loader)
                    self.val_loss_history.append(avg_val_loss)

                Logger.log_info(f"Epoch [{epoch + 1}/{num_epochs}], " +
                                f"Train Loss: {avg_train_loss:.4f} " +
                                f"Validation Loss: {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_improvement = 0
                    Logger.log_info("Validation loss improved " +
                                    f"to {best_val_loss:.4f}.")
                else:
                    epochs_improvement += 1
                    Logger.log_info("No improvement in validation " +
                                    f"loss for {epochs_improvement} epoch(s).")

                # stop training if validation loss does not improve
                if epochs_improvement >= patience:
                    Logger.log_info("Early stopping triggered after " +
                                    f"{patience} epochs without improvement.")
                    break

        except Exception as e:
            Logger.log_error(f"Error while training the model: {str(e)}")
            return None

    def get_model(self) -> nn.Module:
        return self.model

    def get_loss_history(self) -> List[float]:
        return self.train_loss_history, self.val_loss_history


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int, output_size: int,
                 dropout_prob: float = 0.2) -> None:
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        Logger.log_info("Model initialized.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_state = torch.zeros(self.num_layers,
                                   x.size(0), self.hidden_size).to(x.device)
        cell_state = torch.zeros(self.num_layers,
                                 x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (hidden_state, cell_state))
        out = self.fc(out[:, -1, :])
        return out


class ModelEvaluator:
    def __init__(self, model: nn.Module, test_loader: DataLoader) -> None:
        self.model = model
        self.test_loader = test_loader
        self.predictions = []
        self.actuals = []
        Logger.log_info("Evaluation procedure initialized.")

    def evaluate(self) -> None:
        """Evaluate the model."""
        try:
            self.model.eval()
            with torch.no_grad():
                for batch_X, batch_y in self.test_loader:
                    outputs = self.model(batch_X)
                    self.predictions.append(outputs.cpu().numpy())
                    self.actuals.append(batch_y.cpu().numpy())
            self.predictions = np.concatenate(self.predictions, axis=0)
            self.actuals = np.concatenate(self.actuals, axis=0)
            Logger.log_info("Evaluation procedure performed successfully.")

        except Exception as e:
            Logger.log_error(f"Error while evaluating the model: {str(e)}")
            return None

    def calculate_metrics(self) -> None:
        """Calculate the metrics of the model."""
        mse = np.mean((self.actuals - self.predictions) ** 2)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.actuals, self.predictions)

        Logger.log_info(f"Mean Squared Error (MSE): {mse:.4f}")
        Logger.log_info(f"Root Mean Square Error (RMSE): {rmse:.4f}")
        Logger.log_info(f"R2 score: {r2:.4f}")

    def plot_predictions(self) -> None:
        """Plot the predicted against the actual values."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.actuals, label='Actual', alpha=0.7)
        plt.plot(self.predictions, label='Predicted', alpha=0.7)
        plt.title("Predicted vs Actual on Test Data")
        plt.legend()
        plt.show()

    def plot_loss(self, train_loss: List[float],
                  val_loss: List[float]) -> None:
        """Plot the training and validation loss for observation."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='Training loss')
        plt.plot(val_loss, label='Validation loss')
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.show()
