"""FAISS Vector Store for efficient similarity search"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS-based vector store for efficient similarity search.

    Features:
    - L2 and Inner Product (cosine) similarity
    - Persistent storage and loading
    - Metadata support for stored vectors
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",  # "flat", "ivf", "hnsw"
        metric: str = "cosine",  # "cosine", "l2"
        index_path: Path | None = None,
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.index_path = Path(index_path) if index_path else None
        self._index = None
        self._metadata: list[dict[str, Any]] = []
        self._id_to_idx: dict[str, int] = {}

    def _lazy_load(self):
        """Lazy load FAISS and create index"""
        if self._index is None:
            try:
                import faiss

                # Create appropriate index based on type
                if self.metric == "cosine":
                    # For cosine similarity, use inner product with normalized vectors
                    if self.index_type == "flat":
                        self._index = faiss.IndexFlatIP(self.dimension)
                    elif self.index_type == "ivf":
                        quantizer = faiss.IndexFlatIP(self.dimension)
                        self._index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
                    else:
                        self._index = faiss.IndexFlatIP(self.dimension)
                else:
                    # L2 distance
                    if self.index_type == "flat":
                        self._index = faiss.IndexFlatL2(self.dimension)
                    else:
                        self._index = faiss.IndexFlatL2(self.dimension)

                logger.info(
                    f"Created FAISS index: {self.index_type}, metric: {self.metric}"
                )

                # Try to load existing index
                if self.index_path and self.index_path.exists():
                    self.load()

            except ImportError as e:
                logger.error(f"FAISS not installed: {e}")
                raise ImportError(
                    "Please install FAISS: pip install faiss-cpu or faiss-gpu"
                )

    @property
    def index(self):
        """Get the FAISS index, creating it if necessary"""
        self._lazy_load()
        return self._index

    def add(
        self, vectors: np.ndarray, ids: list[str], metadata: list[dict] | None = None
    ):
        """
        Add vectors to the index.

        Args:
            vectors: numpy array of shape (n, dimension)
            ids: List of unique string IDs
            metadata: Optional list of metadata dicts
        """
        if len(vectors) == 0:
            return

        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected dimension {self.dimension}, got {vectors.shape[1]}"
            )

        # Normalize for cosine similarity
        if self.metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / (norms + 1e-8)

        # Add to index
        start_idx = len(self._metadata)
        self.index.add(vectors)

        # Store metadata and ID mapping
        for i, (id_, meta) in enumerate(zip(ids, metadata or [{}] * len(ids))):
            idx = start_idx + i
            self._id_to_idx[id_] = idx
            self._metadata.append({"id": id_, **meta})

        logger.info(
            f"Added {len(vectors)} vectors to index. Total: {self.index.ntotal}"
        )

    def search(
        self, query_vector: np.ndarray, top_k: int = 5, threshold: float | None = None
    ) -> list[dict]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query vector of shape (dimension,) or (1, dimension)
            top_k: Number of results to return
            threshold: Minimum similarity threshold (for cosine)

        Returns:
            List of dicts with 'id', 'score', and metadata
        """
        query_vector = np.asarray(query_vector, dtype=np.float32)

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Normalize for cosine similarity
        if self.metric == "cosine":
            norm = np.linalg.norm(query_vector)
            query_vector = query_vector / (norm + 1e-8)

        # Search
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # Invalid index
                continue

            if threshold is not None and score < threshold:
                continue

            result = {"score": float(score), **self._metadata[idx]}
            results.append(result)

        return results

    def search_batch(
        self, query_vectors: np.ndarray, top_k: int = 5
    ) -> list[list[dict]]:
        """
        Search for similar vectors in batch.

        Args:
            query_vectors: Query vectors of shape (n, dimension)
            top_k: Number of results per query

        Returns:
            List of result lists
        """
        query_vectors = np.asarray(query_vectors, dtype=np.float32)

        if self.metric == "cosine":
            norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
            query_vectors = query_vectors / (norms + 1e-8)

        scores, indices = self.index.search(
            query_vectors, min(top_k, self.index.ntotal)
        )

        all_results = []
        for batch_scores, batch_indices in zip(scores, indices):
            results = []
            for score, idx in zip(batch_scores, batch_indices):
                if idx < 0:
                    continue
                result = {"score": float(score), **self._metadata[idx]}
                results.append(result)
            all_results.append(results)

        return all_results

    def get_by_id(self, id_: str) -> dict | None:
        """Get metadata by ID"""
        idx = self._id_to_idx.get(id_)
        if idx is not None:
            return self._metadata[idx]
        return None

    def save(self, path: Path | None = None):
        """Save index and metadata to disk"""
        import faiss

        save_path = Path(path) if path else self.index_path
        if save_path is None:
            raise ValueError("No save path specified")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(save_path / "index.faiss"))

        # Save metadata
        with open(save_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": self._metadata,
                    "id_to_idx": self._id_to_idx,
                    "dimension": self.dimension,
                    "metric": self.metric,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(f"Saved index to {save_path}")

    def load(self, path: Path | None = None):
        """Load index and metadata from disk"""
        import faiss

        load_path = Path(path) if path else self.index_path
        if load_path is None:
            raise ValueError("No load path specified")

        if not load_path.exists():
            logger.warning(f"Index path does not exist: {load_path}")
            return

        # Load FAISS index
        index_file = load_path / "index.faiss"
        if index_file.exists():
            self._index = faiss.read_index(str(index_file))

        # Load metadata
        metadata_file = load_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, encoding="utf-8") as f:
                data = json.load(f)
                self._metadata = data["metadata"]
                self._id_to_idx = data["id_to_idx"]

        logger.info(
            f"Loaded index from {load_path}, total vectors: {self.index.ntotal}"
        )

    def clear(self):
        """Clear all vectors and metadata"""
        self._index = None
        self._metadata = []
        self._id_to_idx = {}
        self._lazy_load()

    def __len__(self) -> int:
        """Return number of vectors in index"""
        if self._index is None:
            return 0
        return self.index.ntotal
