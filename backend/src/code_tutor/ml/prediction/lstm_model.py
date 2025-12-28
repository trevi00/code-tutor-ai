"""LSTM Model for Time-Series Learning Prediction"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class LSTMPredictor:
    """
    LSTM model for predicting learning success patterns.

    Predicts:
    - Future success rate based on historical performance
    - Time to solve problems
    - Learning velocity trends

    Architecture:
    - Bidirectional LSTM layers
    - Attention mechanism (optional)
    - Dense output for regression/classification
    """

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        sequence_length: int = 30,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = True,
        device: Optional[str] = None
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.output_size = output_size
        self.dropout = dropout
        self.bidirectional = bidirectional

        self._model = None
        self._device = device
        self._optimizer = None
        self._criterion = None

    def _lazy_load(self):
        """Lazy load PyTorch and create model"""
        if self._model is not None:
            return

        try:
            import torch
            import torch.nn as nn

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            class LSTMNetwork(nn.Module):
                def __init__(
                    self,
                    input_size,
                    hidden_size,
                    num_layers,
                    output_size,
                    dropout,
                    bidirectional
                ):
                    super().__init__()

                    self.hidden_size = hidden_size
                    self.num_layers = num_layers
                    self.bidirectional = bidirectional
                    self.num_directions = 2 if bidirectional else 1

                    # LSTM layers
                    self.lstm = nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=True,
                        dropout=dropout if num_layers > 1 else 0,
                        bidirectional=bidirectional
                    )

                    # Attention layer
                    self.attention = nn.Sequential(
                        nn.Linear(hidden_size * self.num_directions, hidden_size),
                        nn.Tanh(),
                        nn.Linear(hidden_size, 1),
                        nn.Softmax(dim=1)
                    )

                    # Output layers
                    lstm_output_size = hidden_size * self.num_directions
                    self.fc = nn.Sequential(
                        nn.Linear(lstm_output_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_size, output_size)
                    )

                def forward(self, x):
                    # x shape: (batch, seq_len, input_size)
                    lstm_out, _ = self.lstm(x)
                    # lstm_out: (batch, seq_len, hidden_size * num_directions)

                    # Apply attention
                    attention_weights = self.attention(lstm_out)
                    # attention_weights: (batch, seq_len, 1)

                    # Weighted sum
                    context = torch.sum(attention_weights * lstm_out, dim=1)
                    # context: (batch, hidden_size * num_directions)

                    # Output
                    output = self.fc(context)
                    return output

            self._model = LSTMNetwork(
                self.input_size,
                self.hidden_size,
                self.num_layers,
                self.output_size,
                self.dropout,
                self.bidirectional
            ).to(self._device)

            self._criterion = nn.MSELoss()
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)

            logger.info(f"Created LSTM model on {self._device}")

        except ImportError as e:
            logger.error(f"PyTorch not installed: {e}")
            raise ImportError("Please install PyTorch: pip install torch")

    @property
    def model(self):
        """Get the model, creating if needed"""
        self._lazy_load()
        return self._model

    def prepare_sequences(
        self,
        data: np.ndarray,
        target_col: int = -1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for training.

        Args:
            data: Time-series data of shape (n_samples, n_features)
            target_col: Column index for target variable

        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []

        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length, target_col])

        return np.array(X), np.array(y)

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """Perform one training step"""
        import torch

        self._lazy_load()
        self._model.train()

        X_tensor = torch.FloatTensor(X).to(self._device)
        y_tensor = torch.FloatTensor(y).to(self._device)

        if y_tensor.ndim == 1:
            y_tensor = y_tensor.unsqueeze(1)

        self._optimizer.zero_grad()

        predictions = self._model(X_tensor)
        loss = self._criterion(predictions, y_tensor)

        loss.backward()
        self._optimizer.step()

        return loss.item()

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.1,
        early_stopping_patience: int = 10
    ) -> dict:
        """
        Train the LSTM model.

        Args:
            X: Input sequences (n_samples, seq_len, n_features)
            y: Target values (n_samples,)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction for validation
            early_stopping_patience: Patience for early stopping

        Returns:
            Training history
        """
        import torch

        self._lazy_load()

        # Split train/val
        n_val = int(len(X) * validation_split)
        indices = np.random.permutation(len(X))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        n_batches = (len(X_train) + batch_size - 1) // batch_size

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Shuffle training data
            perm = np.random.permutation(len(X_train))
            X_train, y_train = X_train[perm], y_train[perm]

            epoch_loss = 0
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(X_train))

                batch_loss = self.train_step(
                    X_train[start_idx:end_idx],
                    y_train[start_idx:end_idx]
                )
                epoch_loss += batch_loss

            avg_loss = epoch_loss / n_batches
            history["train_loss"].append(avg_loss)

            # Validation
            if n_val > 0:
                val_loss = self._evaluate(X_val, y_val)
                history["val_loss"].append(val_loss)

                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f}"
                )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
            else:
                logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

        return history

    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model on data"""
        import torch

        self._model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self._device)
            y_tensor = torch.FloatTensor(y).to(self._device)

            if y_tensor.ndim == 1:
                y_tensor = y_tensor.unsqueeze(1)

            predictions = self._model(X_tensor)
            loss = self._criterion(predictions, y_tensor)

        return loss.item()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input sequences (n_samples, seq_len, n_features)

        Returns:
            Predictions array
        """
        import torch

        self._lazy_load()
        self._model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self._device)
            predictions = self._model(X_tensor).cpu().numpy()

        return predictions.flatten()

    def predict_next(self, sequence: np.ndarray, steps: int = 7) -> np.ndarray:
        """
        Predict multiple future steps.

        Args:
            sequence: Current sequence (seq_len, n_features)
            steps: Number of future steps to predict

        Returns:
            Array of predictions
        """
        predictions = []
        current_seq = sequence.copy()

        for _ in range(steps):
            # Predict next value
            pred = self.predict(current_seq[np.newaxis, ...])[0]
            predictions.append(pred)

            # Update sequence (shift left, add prediction)
            current_seq = np.roll(current_seq, -1, axis=0)
            current_seq[-1, -1] = pred  # Update target column

        return np.array(predictions)

    def save(self, path: Path):
        """Save model to file"""
        import torch

        self._lazy_load()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "config": {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "sequence_length": self.sequence_length,
                "output_size": self.output_size,
                "dropout": self.dropout,
                "bidirectional": self.bidirectional
            }
        }, path)

        logger.info(f"Saved LSTM model to {path}")

    def load(self, path: Path):
        """Load model from file"""
        import torch

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        checkpoint = torch.load(path, map_location=self._device or "cpu")

        config = checkpoint["config"]
        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.sequence_length = config["sequence_length"]
        self.output_size = config["output_size"]
        self.dropout = config["dropout"]
        self.bidirectional = config["bidirectional"]

        self._lazy_load()

        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(f"Loaded LSTM model from {path}")
