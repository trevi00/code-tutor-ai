"""Code Embedder using CodeBERT/DistilCodeBERT for code similarity"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class CodeEmbedder:
    """
    Code embedding using CodeBERT or DistilCodeBERT.

    Specialized for programming language code:
    - Python, JavaScript, Java, etc.
    - 768-dimensional embeddings (CodeBERT)
    - Understands code syntax and semantics
    """

    # Supported languages for code processing
    SUPPORTED_LANGUAGES = [
        "python",
        "javascript",
        "java",
        "c",
        "cpp",
        "go",
        "ruby",
        "php",
    ]

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        device: str | None = None,
        cache_dir: str | None = None,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._device = device

    def _lazy_load(self):
        """Lazy load the model to save memory until needed"""
        if self._model is None:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer

                if self._device is None:
                    self._device = "cuda" if torch.cuda.is_available() else "cpu"

                logger.info(
                    f"Loading code embedding model: {self.model_name} on {self._device}"
                )

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, cache_dir=self.cache_dir
                )
                self._model = AutoModel.from_pretrained(
                    self.model_name, cache_dir=self.cache_dir
                ).to(self._device)
                self._model.eval()

                logger.info("Code embedding model loaded successfully")
            except ImportError as e:
                logger.error(f"transformers not installed: {e}")
                raise ImportError(
                    "Please install transformers: pip install transformers torch"
                )

    @property
    def model(self):
        """Get the model, loading it if necessary"""
        self._lazy_load()
        return self._model

    @property
    def tokenizer(self):
        """Get the tokenizer, loading it if necessary"""
        self._lazy_load()
        return self._tokenizer

    @property
    def embedding_dimension(self) -> int:
        """Return embedding dimension"""
        return 768  # CodeBERT produces 768-dim embeddings

    def _preprocess_code(self, code: str, language: str = "python") -> str:
        """
        Preprocess code for embedding.

        Args:
            code: Source code string
            language: Programming language

        Returns:
            Preprocessed code string
        """
        # Remove excessive whitespace while preserving structure
        lines = code.strip().split("\n")
        lines = [line.rstrip() for line in lines]

        # Add language token for better understanding
        if language.lower() in self.SUPPORTED_LANGUAGES:
            return f"# {language}\n" + "\n".join(lines)
        return "\n".join(lines)

    def embed(self, code: str, language: str = "python") -> np.ndarray:
        """
        Embed a single code snippet.

        Args:
            code: Source code to embed
            language: Programming language of the code

        Returns:
            numpy array of shape (embedding_dimension,)
        """
        return self.embed_batch([code], [language])[0]

    def embed_batch(
        self,
        codes: list[str],
        languages: list[str] | None = None,
        batch_size: int = 8,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Embed a batch of code snippets.

        Args:
            codes: List of source codes to embed
            languages: List of programming languages (defaults to "python")
            batch_size: Batch size for processing
            normalize: Whether to L2-normalize embeddings

        Returns:
            numpy array of shape (len(codes), embedding_dimension)
        """
        import torch

        if not codes:
            return np.array([])

        if languages is None:
            languages = ["python"] * len(codes)

        # Preprocess all codes
        processed = [
            self._preprocess_code(code, lang) for code, lang in zip(codes, languages)
        ]

        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(processed), batch_size):
                batch = processed[i : i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self._device)

                # Get embeddings
                outputs = self.model(**inputs)

                # Use [CLS] token embedding as code representation
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                if normalize:
                    # L2 normalize
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    embeddings = embeddings / (norms + 1e-8)

                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def similarity(self, code1: str, code2: str, language: str = "python") -> float:
        """
        Compute cosine similarity between two code snippets.

        Args:
            code1: First code snippet
            code2: Second code snippet
            language: Programming language

        Returns:
            Cosine similarity score between 0 and 1
        """
        emb1 = self.embed(code1, language)
        emb2 = self.embed(code2, language)

        # Cosine similarity (embeddings are already normalized)
        return float(np.dot(emb1, emb2))

    def find_similar_patterns(
        self,
        query_code: str,
        pattern_codes: list[str],
        pattern_names: list[str],
        language: str = "python",
        top_k: int = 3,
    ) -> list[dict]:
        """
        Find most similar algorithm patterns for given code.

        Args:
            query_code: Query code snippet
            pattern_codes: List of pattern example codes
            pattern_names: List of pattern names
            language: Programming language
            top_k: Number of top results to return

        Returns:
            List of dicts with pattern info and similarity scores
        """
        query_emb = self.embed(query_code, language)
        pattern_embs = self.embed_batch(pattern_codes, [language] * len(pattern_codes))

        # Compute similarities
        similarities = np.dot(pattern_embs, query_emb)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            {
                "pattern": pattern_names[idx],
                "similarity": float(similarities[idx]),
                "rank": i + 1,
            }
            for i, idx in enumerate(top_indices)
        ]

    def unload(self):
        """Unload model from memory"""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None

            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Code embedding model unloaded")
