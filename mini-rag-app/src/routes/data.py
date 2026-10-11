from fastapi import APIRouter, UploadFile, Depends, status, Request
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from celery.result import AsyncResult
from celery_app import celery_app
from tasks.file_processing import process_project_files
from tasks.process_workflow import process_and_push_workflow
from helpers.config import get_settings, Settings
from controllers.DataController import DataController
from controllers.ProjectController import ProjectController
from models.ProjectModel import ProjectModel
from models.AssetModel import AssetModel
from models.dbschemas import Asset
from models.enums.AssetTypeEnum import AssetTypeEnum
from models import ResponseSignal
from .scheams.data import processingRequest
import aiofiles
import logging
import os

logger = logging.getLogger("uvicorn.error")

data_router = APIRouter()

data_controller = DataController()
project_controller = ProjectController()


@data_router.post("/upload/{project_id}")
async def upload_data(request: Request, project_id: int, file: UploadFile,
                      app_settings: Settings = Depends(get_settings)):

    is_valid, result_signal = data_controller.validate_uploaded_file(file=file)

    if not is_valid:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                            content={"signal": result_signal.value})

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    file_path, file_id = data_controller.generate_unique_filepath(
        orig_file_name=file.filename, project_id=project_id)

    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    except Exception as e:
        logger.error(f"Error while uploading file: {e}")
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                            content={"signal": ResponseSignal.FILE_UPLOAD_FAILED.value})

    asset_model = await AssetModel.create_instance(db_client=request.app.db_client)

    asset_resource = Asset(
        asset_project_id=project.project_id,
        asset_type=AssetTypeEnum.FILE.value,
        asset_name=file_id,
        asset_size=os.path.getsize(file_path),
        asset_config={
            "orig_name": file.filename,
            "content_type": file.content_type,
        },
    )

    asset_record = await asset_model.create_asset(asset=asset_resource)

    return JSONResponse(content={
        "signal": ResponseSignal.FILE_UPLOAD_SUCCESS.value,
        "file_id": file_id,
        "asset_id": str(asset_record.asset_id),
    })


@data_router.post("/process/{project_id}")
async def process_data(project_id: int, process_request: processingRequest):

    task = process_project_files.delay(
        project_id=project_id,
        file_id=process_request.file_id,
        chunk_size=process_request.chunk_size,
        overlap_size=process_request.overlap_size,
        do_reset=process_request.do_reset,
    )

    return JSONResponse(content={
        "signal": ResponseSignal.PROCESSING_QUEUED.value,
        "task_id": task.id,
    })


@data_router.post("/process-and-push/{project_id}")
async def process_and_push(project_id: int, process_request: processingRequest):

    workflow_task = process_and_push_workflow.delay(
        project_id=project_id,
        file_id=process_request.file_id,
        chunk_size=process_request.chunk_size,
        overlap_size=process_request.overlap_size,
        do_reset=process_request.do_reset,
    )

    return JSONResponse(content={
        "signal": ResponseSignal.WORKFLOW_QUEUED.value,
        "workflow_task_id": workflow_task.id,
    })


@data_router.get("/task/{task_id}")
async def get_task_status(task_id: str):

    result = AsyncResult(task_id, app=celery_app)
    task_result = None

    if result.ready():
        task_result = result.result if result.successful() else {"error": str(result.result)}

    return JSONResponse(content={
        "signal": ResponseSignal.TASK_STATUS_RETRIEVED.value,
        "task_id": task_id,
        "status": result.status,
        "result": jsonable_encoder(task_result),
    })
