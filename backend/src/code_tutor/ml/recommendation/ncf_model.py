"""Neural Collaborative Filtering Model for Problem Recommendation"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class NCFModel:
    """
    Neural Collaborative Filtering Model.

    Architecture:
    - User embedding layer
    - Item (problem) embedding layer
    - GMF (Generalized Matrix Factorization) pathway
    - MLP (Multi-Layer Perceptron) pathway
    - Combined output for prediction

    Based on "Neural Collaborative Filtering" (He et al., 2017)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        hidden_layers: List[int] = None,
        dropout: float = 0.2,
        device: Optional[str] = None
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers or [64, 32, 16]
        self.dropout = dropout

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

            # Define the NCF model
            class NCFNetwork(nn.Module):
                def __init__(self, num_users, num_items, embedding_dim, hidden_layers, dropout):
                    super().__init__()

                    # GMF embeddings
                    self.gmf_user_embedding = nn.Embedding(num_users, embedding_dim)
                    self.gmf_item_embedding = nn.Embedding(num_items, embedding_dim)

                    # MLP embeddings
                    self.mlp_user_embedding = nn.Embedding(num_users, embedding_dim)
                    self.mlp_item_embedding = nn.Embedding(num_items, embedding_dim)

                    # MLP layers
                    mlp_layers = []
                    input_size = embedding_dim * 2
                    for hidden_size in hidden_layers:
                        mlp_layers.extend([
                            nn.Linear(input_size, hidden_size),
                            nn.ReLU(),
                            nn.Dropout(dropout)
                        ])
                        input_size = hidden_size
                    self.mlp = nn.Sequential(*mlp_layers)

                    # Final prediction layer
                    # GMF output (embedding_dim) + MLP output (last hidden layer)
                    self.predict_layer = nn.Linear(embedding_dim + hidden_layers[-1], 1)
                    self.sigmoid = nn.Sigmoid()

                    self._init_weights()

                def _init_weights(self):
                    """Initialize weights with Xavier initialization"""
                    for module in self.modules():
                        if isinstance(module, nn.Embedding):
                            nn.init.xavier_uniform_(module.weight)
                        elif isinstance(module, nn.Linear):
                            nn.init.xavier_uniform_(module.weight)
                            if module.bias is not None:
                                nn.init.zeros_(module.bias)

                def forward(self, user_ids, item_ids):
                    # GMF pathway
                    gmf_user = self.gmf_user_embedding(user_ids)
                    gmf_item = self.gmf_item_embedding(item_ids)
                    gmf_output = gmf_user * gmf_item  # Element-wise product

                    # MLP pathway
                    mlp_user = self.mlp_user_embedding(user_ids)
                    mlp_item = self.mlp_item_embedding(item_ids)
                    mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
                    mlp_output = self.mlp(mlp_input)

                    # Combine GMF and MLP
                    combined = torch.cat([gmf_output, mlp_output], dim=-1)

                    # Predict
                    output = self.predict_layer(combined)
                    return self.sigmoid(output).squeeze()

            self._model = NCFNetwork(
                self.num_users,
                self.num_items,
                self.embedding_dim,
                self.hidden_layers,
                self.dropout
            ).to(self._device)

            self._criterion = nn.BCELoss()
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)

            logger.info(f"Created NCF model on {self._device}")

        except ImportError as e:
            logger.error(f"PyTorch not installed: {e}")
            raise ImportError("Please install PyTorch: pip install torch")

    @property
    def model(self):
        """Get the model, creating if needed"""
        self._lazy_load()
        return self._model

    def train_step(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Perform one training step.

        Args:
            user_ids: Array of user IDs
            item_ids: Array of item (problem) IDs
            labels: Binary labels (1 = solved, 0 = not solved)

        Returns:
            Loss value
        """
        import torch

        self._lazy_load()
        self._model.train()

        user_tensor = torch.LongTensor(user_ids).to(self._device)
        item_tensor = torch.LongTensor(item_ids).to(self._device)
        label_tensor = torch.FloatTensor(labels).to(self._device)

        self._optimizer.zero_grad()

        predictions = self._model(user_tensor, item_tensor)
        loss = self._criterion(predictions, label_tensor)

        loss.backward()
        self._optimizer.step()

        return loss.item()

    def train(
        self,
        train_data: List[Tuple[int, int, int]],
        val_data: Optional[List[Tuple[int, int, int]]] = None,
        epochs: int = 20,
        batch_size: int = 256,
        early_stopping_patience: int = 5
    ) -> dict:
        """
        Train the NCF model.

        Args:
            train_data: List of (user_id, item_id, label) tuples
            val_data: Optional validation data
            epochs: Number of training epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping

        Returns:
            Training history dict
        """
        import torch

        self._lazy_load()

        train_array = np.array(train_data)
        user_ids = train_array[:, 0]
        item_ids = train_array[:, 1]
        labels = train_array[:, 2]

        n_samples = len(train_data)
        n_batches = (n_samples + batch_size - 1) // batch_size

        history = {"train_loss": [], "val_loss": [], "val_auc": []}
        best_val_auc = 0
        patience_counter = 0

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            epoch_loss = 0

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                batch_loss = self.train_step(
                    user_ids[batch_indices],
                    item_ids[batch_indices],
                    labels[batch_indices]
                )
                epoch_loss += batch_loss

            avg_loss = epoch_loss / n_batches
            history["train_loss"].append(avg_loss)

            # Validation
            if val_data:
                val_loss, val_auc = self._evaluate(val_data)
                history["val_loss"].append(val_loss)
                history["val_auc"].append(val_auc)

                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f} - Val AUC: {val_auc:.4f}"
                )

                # Early stopping
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
            else:
                logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

        return history

    def _evaluate(self, data: List[Tuple[int, int, int]]) -> Tuple[float, float]:
        """Evaluate model on data"""
        import torch
        from sklearn.metrics import roc_auc_score

        self._model.eval()

        data_array = np.array(data)
        user_ids = torch.LongTensor(data_array[:, 0]).to(self._device)
        item_ids = torch.LongTensor(data_array[:, 1]).to(self._device)
        labels = data_array[:, 2]

        with torch.no_grad():
            predictions = self._model(user_ids, item_ids).cpu().numpy()

        loss = float(self._criterion(
            torch.FloatTensor(predictions),
            torch.FloatTensor(labels)
        ))

        try:
            auc = roc_auc_score(labels, predictions)
        except:
            auc = 0.5

        return loss, auc

    def predict(
        self,
        user_id: int,
        item_ids: np.ndarray
    ) -> np.ndarray:
        """
        Predict scores for items.

        Args:
            user_id: User ID
            item_ids: Array of item IDs to score

        Returns:
            Array of prediction scores
        """
        import torch

        self._lazy_load()
        self._model.eval()

        user_tensor = torch.LongTensor([user_id] * len(item_ids)).to(self._device)
        item_tensor = torch.LongTensor(item_ids).to(self._device)

        with torch.no_grad():
            predictions = self._model(user_tensor, item_tensor).cpu().numpy()

        return predictions

    def recommend(
        self,
        user_id: int,
        candidate_items: Optional[np.ndarray] = None,
        top_k: int = 10,
        exclude_items: Optional[set] = None
    ) -> List[Tuple[int, float]]:
        """
        Recommend items for a user.

        Args:
            user_id: User ID
            candidate_items: Items to consider (all items if None)
            top_k: Number of recommendations
            exclude_items: Items to exclude (e.g., already solved)

        Returns:
            List of (item_id, score) tuples
        """
        if candidate_items is None:
            candidate_items = np.arange(self.num_items)

        if exclude_items:
            candidate_items = np.array([
                item for item in candidate_items
                if item not in exclude_items
            ])

        if len(candidate_items) == 0:
            return []

        scores = self.predict(user_id, candidate_items)

        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            (int(candidate_items[idx]), float(scores[idx]))
            for idx in top_indices
        ]

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
                "num_users": self.num_users,
                "num_items": self.num_items,
                "embedding_dim": self.embedding_dim,
                "hidden_layers": self.hidden_layers,
                "dropout": self.dropout
            }
        }, path)

        logger.info(f"Saved NCF model to {path}")

    def load(self, path: Path):
        """Load model from file"""
        import torch

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        checkpoint = torch.load(path, map_location=self._device or "cpu")

        config = checkpoint["config"]
        self.num_users = config["num_users"]
        self.num_items = config["num_items"]
        self.embedding_dim = config["embedding_dim"]
        self.hidden_layers = config["hidden_layers"]
        self.dropout = config["dropout"]

        self._lazy_load()

        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(f"Loaded NCF model from {path}")

    def get_user_embedding(self, user_id: int) -> np.ndarray:
        """Get user embedding vector"""
        import torch

        self._lazy_load()
        self._model.eval()

        user_tensor = torch.LongTensor([user_id]).to(self._device)

        with torch.no_grad():
            gmf_emb = self._model.gmf_user_embedding(user_tensor)
            mlp_emb = self._model.mlp_user_embedding(user_tensor)
            combined = torch.cat([gmf_emb, mlp_emb], dim=-1)

        return combined.cpu().numpy().flatten()

    def get_item_embedding(self, item_id: int) -> np.ndarray:
        """Get item embedding vector"""
        import torch

        self._lazy_load()
        self._model.eval()

        item_tensor = torch.LongTensor([item_id]).to(self._device)

        with torch.no_grad():
            gmf_emb = self._model.gmf_item_embedding(item_tensor)
            mlp_emb = self._model.mlp_item_embedding(item_tensor)
            combined = torch.cat([gmf_emb, mlp_emb], dim=-1)

        return combined.cpu().numpy().flatten()
