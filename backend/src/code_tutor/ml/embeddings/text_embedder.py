"""Text Embedder using Sentence Transformers for Korean/English text"""

import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class TextEmbedder:
    """
    Text embedding using sentence-transformers.
    Supports multilingual text including Korean.

    Uses paraphrase-multilingual-MiniLM-L12-v2 by default for:
    - 384-dimensional embeddings
    - Multi-language support (Korean, English, etc.)
    - Fast inference on CPU/GPU
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None
        self._device = device

    def _lazy_load(self):
        """Lazy load the model to save memory until needed"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                import torch

                if self._device is None:
                    self._device = "cuda" if torch.cuda.is_available() else "cpu"

                logger.info(f"Loading text embedding model: {self.model_name} on {self._device}")
                self._model = SentenceTransformer(
                    self.model_name,
                    cache_folder=self.cache_dir,
                    device=self._device
                )
                logger.info("Text embedding model loaded successfully")
            except ImportError as e:
                logger.error(f"sentence-transformers not installed: {e}")
                raise ImportError(
                    "Please install sentence-transformers: pip install sentence-transformers"
                )

    @property
    def model(self):
        """Get the model, loading it if necessary"""
        self._lazy_load()
        return self._model

    @property
    def embedding_dimension(self) -> int:
        """Return embedding dimension"""
        return 384  # MiniLM-L12 produces 384-dim embeddings

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string.

        Args:
            text: Input text to embed

        Returns:
            numpy array of shape (embedding_dimension,)
        """
        return self.embed_batch([text])[0]

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Embed a batch of text strings.

        Args:
            texts: List of input texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            normalize: Whether to L2-normalize embeddings

        Returns:
            numpy array of shape (len(texts), embedding_dimension)
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )

        return embeddings

    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score between 0 and 1
        """
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)

        # Cosine similarity (embeddings are already normalized)
        return float(np.dot(emb1, emb2))

    def find_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar texts from candidates.

        Args:
            query: Query text
            candidates: List of candidate texts to search
            top_k: Number of top results to return

        Returns:
            List of (index, score) tuples sorted by similarity
        """
        query_emb = self.embed(query)
        candidate_embs = self.embed_batch(candidates)

        # Compute similarities
        similarities = np.dot(candidate_embs, query_emb)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def unload(self):
        """Unload model from memory"""
        if self._model is not None:
            del self._model
            self._model = None

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Text embedding model unloaded")
