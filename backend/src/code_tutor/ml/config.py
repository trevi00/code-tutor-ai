"""ML Configuration"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class MLConfig(BaseSettings):
    """Machine Learning configuration"""

    # Model paths
    MODEL_CACHE_DIR: Path = Field(default=Path("./models"))
    FAISS_INDEX_PATH: Path = Field(default=Path("./data/faiss_index"))
    PATTERN_DATA_PATH: Path = Field(default=Path("./data/patterns.json"))

    # LLM Settings (EEVE-Korean or OpenAI fallback)
    USE_LOCAL_LLM: bool = Field(default=True)
    LOCAL_LLM_MODEL: str = Field(default="yanolja/EEVE-Korean-2.8B-v1.0")
    OPENAI_API_KEY: str = Field(default="")
    OPENAI_MODEL: str = Field(default="gpt-3.5-turbo")

    # Embedding Settings
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    EMBEDDING_DIMENSION: int = Field(default=384)

    # CodeBERT Settings
    CODE_EMBEDDING_MODEL: str = Field(default="microsoft/codebert-base")
    CODE_EMBEDDING_DIMENSION: int = Field(default=768)

    # NCF Settings
    NCF_EMBEDDING_DIM: int = Field(default=32)
    NCF_HIDDEN_LAYERS: list[int] = Field(default=[64, 32, 16])
    NCF_MODEL_PATH: Path = Field(default=Path("./models/ncf_model.pt"))

    # LSTM Settings
    LSTM_HIDDEN_SIZE: int = Field(default=64)
    LSTM_NUM_LAYERS: int = Field(default=2)
    LSTM_SEQUENCE_LENGTH: int = Field(default=30)
    LSTM_MODEL_PATH: Path = Field(default=Path("./models/lstm_model.pt"))

    # RAG Settings
    RAG_TOP_K: int = Field(default=3)
    RAG_SIMILARITY_THRESHOLD: float = Field(default=0.1)  # Lowered for Korean queries

    # Hardware Settings
    USE_GPU: bool = Field(default=True)
    MAX_GPU_MEMORY_GB: float = Field(default=4.0)
    QUANTIZATION_BITS: int = Field(default=4)

    class Config:
        env_prefix = "ML_"
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_ml_config() -> MLConfig:
    """Get ML configuration singleton"""
    return MLConfig()
