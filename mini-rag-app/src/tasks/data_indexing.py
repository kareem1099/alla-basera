from celery_app import celery_app, get_setup_utils, is_retryable_error, TaskError
from helpers.config import get_settings
from controllers.NLPController import NLPController
from models.ProjectModel import ProjectModel
from models.DataChunckmodel import ChunkModel
from models import ResponseSignal
from utils.idempotency_manager import IdempotencyManager
import asyncio
import logging

logger = logging.getLogger("celery.task")

TASK_NAME = "tasks.data_indexing.index_data_content"


@celery_app.task(bind=True, name=TASK_NAME, max_retries=5,
                 time_limit=get_settings().CELERY_INDEXING_TIME_LIMIT)
def index_data_content(self, project_id: int, do_reset: int):
    try:
        return asyncio.run(_index_data_content(self, project_id, do_reset, TASK_NAME))
    except Exception as error:
        if is_retryable_error(error):
            raise self.retry(exc=error, countdown=60)
        raise


async def _index_data_content(task_instance, project_id: int, do_reset: int, task_name: str):
    settings = get_settings()
    db_engine, vectordb_client = None, None

    try:
        db_engine, db_client, embedding_client, vectordb_client = await get_setup_utils()

        idempotency_manager = IdempotencyManager(db_client, db_engine)
        task_args = {"project_id": project_id, "do_reset": do_reset}

        should_execute, execution = await idempotency_manager.should_execute_task(
            task_name=task_name,
            task_args=task_args,
            celery_task_id=task_instance.request.id,
            task_time_limit=settings.CELERY_INDEXING_TIME_LIMIT,
        )

        if not should_execute:
            logger.warning(f"Task already running | execution_id: {execution.execution_id}")
            return {
                "signal": ResponseSignal.TASK_ALREADY_RUNNING.value,
                "running_task_id": str(execution.celery_task_id),
            }

        is_retry = bool(task_instance.request.retries)

        try:
            result = await push_project(
                db_client=db_client,
                embedding_client=embedding_client,
                vectordb_client=vectordb_client,
                project_id=project_id,
                do_reset=bool(do_reset) and not is_retry,
                batch_delay=settings.CELERY_INDEXING_BATCH_DELAY,
            )
        except Exception as error:
            await idempotency_manager.update_task_status(
                execution.execution_id, "FAILURE", {"error": str(error)})
            raise

        await idempotency_manager.update_task_status(execution.execution_id, "SUCCESS", result)
        return result

    finally:
        if db_engine:
            await db_engine.dispose()
        if vectordb_client:
            await vectordb_client.disconnect()


async def push_project(db_client, embedding_client, vectordb_client, project_id: int,
                       do_reset: bool, batch_delay: float = 0):

    project_model = await ProjectModel.create_instance(db_client=db_client)
    chunk_model = await ChunkModel.create_instance(db_client=db_client)

    project = await project_model.get_project_or_create_one(project_id=project_id)

    if not project:
        raise TaskError(ResponseSignal.PROJECT_NOT_FOUND_ERROR.value)

    nlp_controller = NLPController(
        vectordb_client=vectordb_client,
        generation_client=None,
        embedding_client=embedding_client,
        template_parser=None,
    )

    _ = await nlp_controller.create_vector_db_collection(project=project, do_reset=do_reset)

    indexed_chunk_ids = await nlp_controller.get_indexed_chunk_ids(project=project)

    page_no = 1
    inserted_items_count = 0
    skipped_items_count = 0

    while True:
        page_chunks = await chunk_model.get_project_chunks(project_id=project.project_id,
                                                           page_no=page_no)

        if not page_chunks:
            break

        page_no += 1

        new_chunks = [c for c in page_chunks if c.chunk_id not in indexed_chunk_ids]
        skipped_items_count += len(page_chunks) - len(new_chunks)

        if not new_chunks:
            continue

        if inserted_items_count and batch_delay:
            await asyncio.sleep(batch_delay)

        is_inserted = await nlp_controller.index_into_vector_db(
            project=project,
            chunks=new_chunks,
            chunks_ids=[c.chunk_id for c in new_chunks],
        )

        if not is_inserted:
            raise TaskError(ResponseSignal.INSERT_INTO_VECTORDB_ERROR.value)

        inserted_items_count += len(new_chunks)

    is_index_created = await nlp_controller.create_vector_db_index(project=project)
    is_keyword_stats_refreshed = await nlp_controller.refresh_keyword_stats(project=project)

    return {
        "signal": ResponseSignal.INSERT_INTO_VECTORDB_SUCCESS.value,
        "project_id": project_id,
        "inserted_items_count": inserted_items_count,
        "skipped_items_count": skipped_items_count,
        "is_index_created": bool(is_index_created),
        "is_keyword_stats_refreshed": bool(is_keyword_stats_refreshed),
    }
