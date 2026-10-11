from celery_app import celery_app, get_setup_utils
from helpers.config import get_settings
from utils.idempotency_manager import IdempotencyManager
import asyncio
import logging

logger = logging.getLogger("celery.task")

TASK_NAME = "tasks.maintenance.clean_celery_executions_table"


@celery_app.task(bind=True, name=TASK_NAME, max_retries=1)
def clean_celery_executions_table(self):
    return asyncio.run(_clean_celery_executions_table(self))


async def _clean_celery_executions_table(task_instance):
    settings = get_settings()
    db_engine, vectordb_client = None, None

    try:
        db_engine, db_client, _, vectordb_client = await get_setup_utils()

        idempotency_manager = IdempotencyManager(db_client, db_engine)
        deleted_count = await idempotency_manager.cleanup_old_tasks(
            time_retention=settings.CELERY_EXECUTIONS_RETENTION_SECONDS)

        logger.info(f"Cleaned {deleted_count} old task execution records")
        return {"deleted_count": deleted_count}

    finally:
        if db_engine:
            await db_engine.dispose()
        if vectordb_client:
            await vectordb_client.disconnect()
