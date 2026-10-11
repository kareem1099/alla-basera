from celery import chain
from celery_app import celery_app, is_retryable_error
from helpers.config import get_settings
from tasks.file_processing import process_project_files
from tasks.data_indexing import _index_data_content
from models import ResponseSignal
import asyncio
import logging

logger = logging.getLogger("celery.task")

PUSH_TASK_NAME = "tasks.process_workflow.push_after_process_task"
WORKFLOW_TASK_NAME = "tasks.process_workflow.process_and_push_workflow"


@celery_app.task(bind=True, name=PUSH_TASK_NAME, max_retries=5,
                 time_limit=get_settings().CELERY_INDEXING_TIME_LIMIT)
def push_after_process_task(self, prev_task_result: dict):

    if not prev_task_result or prev_task_result.get("signal") != ResponseSignal.PROCESSING_SUCCESS.value:
        logger.warning(f"Skipping push, processing did not succeed: {prev_task_result}")
        return {"signal": ResponseSignal.PROCESSING_FAILED.value, "processing_result": prev_task_result}

    project_id = prev_task_result.get("project_id")
    do_reset = prev_task_result.get("do_reset")

    try:
        task_results = asyncio.run(_index_data_content(self, project_id, do_reset, PUSH_TASK_NAME))
    except Exception as error:
        if is_retryable_error(error):
            raise self.retry(exc=error, countdown=60)
        raise

    return {
        "project_id": project_id,
        "do_reset": do_reset,
        "processing_result": prev_task_result,
        "task_results": task_results,
    }


@celery_app.task(bind=True, name=WORKFLOW_TASK_NAME)
def process_and_push_workflow(self, project_id: int, file_id: str, chunk_size: int,
                              overlap_size: int, do_reset: int):

    workflow = chain(
        process_project_files.s(project_id, file_id, chunk_size, overlap_size, do_reset),
        push_after_process_task.s(),
    )

    result = workflow.apply_async()

    return {
        "signal": ResponseSignal.WORKFLOW_STARTED.value,
        "workflow_id": result.id,
        "tasks": [
            {"name": "process_project_files", "task_id": result.parent.id if result.parent else None},
            {"name": "push_after_process_task", "task_id": result.id},
        ],
    }
