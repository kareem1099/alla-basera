from celery import Celery
from urllib.parse import quote_plus
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError, InterfaceError
from helpers.config import get_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory

settings = get_settings()

RETRYABLE_ERRORS = (ConnectionError, TimeoutError, OperationalError, InterfaceError)
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
RETRYABLE_ERROR_NAMES = {"TooManyRequestsError", "ServiceUnavailableError", "GatewayTimeoutError",
                         "InternalServerError", "ConnectError", "ReadTimeout", "ConnectTimeout"}


class TaskError(Exception):

    def __init__(self, signal: str):
        super().__init__(signal)
        self.signal = signal


def is_retryable_error(error: Exception) -> bool:
    if isinstance(error, TaskError):
        return False
    if isinstance(error, RETRYABLE_ERRORS):
        return True
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        status_code = getattr(getattr(error, "response", None), "status_code", None)
    if status_code in RETRYABLE_STATUS_CODES:
        return True
    return type(error).__name__ in RETRYABLE_ERROR_NAMES


async def get_setup_utils():
    settings = get_settings()

    postgres_conn = (
        f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{quote_plus(settings.POSTGRES_PASSWORD)}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
    )

    db_engine = create_async_engine(
        postgres_conn,
        connect_args={"ssl": "require"} if settings.POSTGRES_SSL else {},
    )
    db_client = sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    llm_provider_factory = LLMProviderFactory(settings)
    embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
    embedding_client.set_embedding_model(model_id=settings.EMBEDDING_MODEL_ID,
                                         embedding_size=settings.EMBEDDING_MODEL_SIZE)

    vectordb_provider_factory = VectorDBProviderFactory(config=settings, db_client=db_client)
    vectordb_client = vectordb_provider_factory.create(provider=settings.VECTOR_DB_BACKEND)
    await vectordb_client.connect()

    return db_engine, db_client, embedding_client, vectordb_client


celery_app = Celery(
    "minirag",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "tasks.file_processing",
        "tasks.data_indexing",
        "tasks.process_workflow",
        "tasks.maintenance",
    ],
)

celery_app.conf.update(
    task_serializer=settings.CELERY_TASK_SERIALIZER,
    result_serializer=settings.CELERY_TASK_SERIALIZER,
    accept_content=[settings.CELERY_TASK_SERIALIZER],

    task_acks_late=settings.CELERY_TASK_ACKS_LATE,
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,

    task_ignore_result=False,
    result_expires=3600,
    result_extended=True,

    worker_concurrency=settings.CELERY_WORKER_CONCURRENCY,
    worker_prefetch_multiplier=1,

    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,
    worker_cancel_long_running_tasks_on_connection_loss=True,

    task_routes={
        "tasks.file_processing.process_project_files": {"queue": "file_processing"},
        "tasks.data_indexing.index_data_content": {"queue": "data_indexing"},
        "tasks.process_workflow.process_and_push_workflow": {"queue": "file_processing"},
        "tasks.process_workflow.push_after_process_task": {"queue": "data_indexing"},
        "tasks.maintenance.clean_celery_executions_table": {"queue": "default"},
    },

    beat_schedule={
        "cleanup-old-task-records": {
            "task": "tasks.maintenance.clean_celery_executions_table",
            "schedule": 3600,
            "args": (),
        }
    },

    timezone="UTC",
)

celery_app.conf.task_default_queue = "default"
