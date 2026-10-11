from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    APP_NAME: str
    VERSION: str
    FILE_ALLOWED_EXTENSIONS: list
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int
    POSTGRES_USERNAME: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_MAIN_DATABASE: str
    POSTGRES_SSL: bool = False

    APP_API_KEY: Optional[str] = None

    GENERATION_BACKEND: str
    EMBEDDING_BACKEND: str

    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_URL: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_THINKING_BUDGET: Optional[int] = None
    GEMINI_FALLBACK_MODELS: Optional[str] = None
    GENERATION_FALLBACK_BACKEND: Optional[str] = None
    GENERATION_FALLBACK_MODEL_ID: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_API_URL: str = "https://openrouter.ai/api/v1"
    QDRANT_API_KEY: Optional[str] = None

    GENERATION_MODEL_ID: Optional[str] = None
    EMBEDDING_MODEL_ID: Optional[str] = None
    EMBEDDING_MODEL_SIZE: Optional[int] = None
    INPUT_DAFAULT_MAX_CHARACTERS: Optional[int] = None
    GENERATION_DAFAULT_MAX_TOKENS: Optional[int] = None
    GENERATION_DAFAULT_TEMPERATURE: Optional[float] = None

    VECTOR_DB_BACKEND: str
    VECTOR_DB_PATH: str
    VECTOR_DB_DISTANCE_METHOD: Optional[str] = None
    VECTOR_DB_PGVEC_INDEX_THRESHOLD: int = 100

    GROUNDED_MAX_OUTPUT_TOKENS: int = 1200
    GROUNDED_FREQUENCY_PENALTY: float = 0.6
    GROUNDED_PRESENCE_PENALTY: float = 0.3
    GROUNDED_TEMPERATURE: float = 0.2
    GROUNDING_MIN_RATIO: float = 0.1
    CONTEXT_WINDOW_CHILDREN: Optional[int] = None
    CONTEXT_MAX_CHILDREN_PER_PARENT: Optional[int] = None

    RERANK_BACKEND: Optional[str] = None
    RERANK_API_URL: Optional[str] = None
    RERANK_API_KEY: Optional[str] = None
    RERANK_MODEL_ID: Optional[str] = None
    RERANK_CANDIDATES: int = 20
    RERANK_TARGET: str = "parent"
    RERANK_MAX_CHARACTERS: int = 4000

    PRIMARY_LANG: str = "en"
    DEFAULT_LANG: str = "en"

    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None
    CELERY_TASK_SERIALIZER: str = "json"
    CELERY_TASK_TIME_LIMIT: int = 600
    CELERY_INDEXING_TIME_LIMIT: int = 1800
    CELERY_TASK_ACKS_LATE: bool = True
    CELERY_WORKER_CONCURRENCY: int = 2
    CELERY_FLOWER_PASSWORD: Optional[str] = None
    CELERY_INDEXING_BATCH_DELAY: float = 0
    CELERY_EXECUTIONS_RETENTION_SECONDS: int = 86400

    class Config:
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    return Settings()
