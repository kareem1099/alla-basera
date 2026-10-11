from celery_app import celery_app, get_setup_utils, is_retryable_error, TaskError
from helpers.config import get_settings
from controllers.ProcessingController import ProcessingController
from models.ProjectModel import ProjectModel
from models.ParentModel import ParentModel
from models.AssetModel import AssetModel
from models.dbschemas import DataChunk, Parent
from models.enums.AssetTypeEnum import AssetTypeEnum
from models import ResponseSignal
from utils.idempotency_manager import IdempotencyManager
import asyncio
import logging

logger = logging.getLogger("celery.task")

TASK_NAME = "tasks.file_processing.process_project_files"


@celery_app.task(bind=True, name=TASK_NAME, max_retries=3)
def process_project_files(self, project_id: int, file_id: str, chunk_size: int,
                          overlap_size: int, do_reset: int):
    try:
        return asyncio.run(
            _process_project_files(self, project_id, file_id, chunk_size, overlap_size, do_reset)
        )
    except Exception as error:
        if is_retryable_error(error):
            raise self.retry(exc=error, countdown=60)
        raise


async def _process_project_files(task_instance, project_id: int, file_id: str, chunk_size: int,
                                 overlap_size: int, do_reset: int):
    settings = get_settings()
    db_engine, vectordb_client = None, None

    try:
        db_engine, db_client, _, vectordb_client = await get_setup_utils()

        idempotency_manager = IdempotencyManager(db_client, db_engine)
        task_args = {"project_id": project_id, "file_id": file_id, "chunk_size": chunk_size,
                     "overlap_size": overlap_size, "do_reset": do_reset}

        should_execute, execution = await idempotency_manager.should_execute_task(
            task_name=TASK_NAME,
            task_args=task_args,
            celery_task_id=task_instance.request.id,
            task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
        )

        if not should_execute:
            logger.warning(f"Task already running | execution_id: {execution.execution_id}")
            return {
                "signal": ResponseSignal.TASK_ALREADY_RUNNING.value,
                "running_task_id": str(execution.celery_task_id),
            }

        try:
            result = await process_files(db_client, project_id, file_id, chunk_size,
                                         overlap_size, do_reset)
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


async def process_files(db_client, project_id: int, file_id: str, chunk_size: int,
                        overlap_size: int, do_reset: int):

    project_model = await ProjectModel.create_instance(db_client=db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    asset_model = await AssetModel.create_instance(db_client=db_client)

    if file_id:
        asset_record = await asset_model.get_asset_record(
            asset_project_id=project.project_id,
            asset_name=file_id,
        )

        if asset_record is None:
            raise TaskError(ResponseSignal.FILE_ID_NOT_FOUND.value)

        project_files_ids = {asset_record.asset_id: asset_record.asset_name}

    else:
        project_files = await asset_model.get_all_project_assets(
            asset_project_id=project.project_id,
            asset_type=AssetTypeEnum.FILE.value,
        )

        project_files_ids = {record.asset_id: record.asset_name for record in project_files}

    if len(project_files_ids) == 0:
        raise TaskError(ResponseSignal.NO_FILES_ERROR.value)

    processing_controller = ProcessingController(project_id=project_id)
    parent_model = await ParentModel.create_instance(db_client=db_client)

    if do_reset == 1:
        _ = await parent_model.delete_parents_by_project_id(project_id=project.project_id)

    no_parents = 0
    no_records = 0
    no_files = 0

    for asset_id, asset_file_id in project_files_ids.items():

        parents = processing_controller.process_file(
            file_id=asset_file_id,
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
        )

        if parents is None or len(parents) == 0:
            raise TaskError(ResponseSignal.PROCESSING_FAILED.value)

        parents_records = [
            Parent(
                parent_text=parent["text"],
                parent_metadata=parent["metadata"],
                parent_order=parent["order"],
                parent_project_id=project.project_id,
                parent_asset_id=asset_id,
            )
            for parent in parents
        ]

        chunk_order = 0
        parents_chunks_records = []
        for parent in parents:
            parent_chunks = []
            for child in parent["children"]:
                chunk_order += 1
                parent_chunks.append(DataChunk(
                    chunk_text=child["text"],
                    chunk_metadata=child["metadata"],
                    chunk_order=chunk_order,
                    chunk_project_id=project.project_id,
                    chunk_asset_id=asset_id,
                ))
            parents_chunks_records.append(parent_chunks)

        inserted_parents, inserted_chunks = await parent_model.insert_parents_with_chunks(
            parents=parents_records,
            parents_chunks=parents_chunks_records,
        )

        no_parents += inserted_parents
        no_records += inserted_chunks
        no_files += 1

    return {
        "signal": ResponseSignal.PROCESSING_SUCCESS.value,
        "project_id": project_id,
        "do_reset": do_reset,
        "inserted_parents": no_parents,
        "inserted_chunks": no_records,
        "processed_files": no_files,
    }
