from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from models.ProjectModel import ProjectModel
from models import ResponseSignal
from controllers.NLPController import NLPController
from utils.metrics import record_rag_answer
from tasks.data_indexing import index_data_content
from .scheams.nlp import PushRequest, SearchRequest
import logging

logger = logging.getLogger("uvicorn.error")

nlp_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["api_v1", "nlp"],
)


def build_nlp_controller(request: Request):
    return NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
        rerank_client=request.app.rerank_client,
        fallback_generation_client=request.app.fallback_generation_client,
    )


@nlp_router.post("/index/push/{project_id}")
async def index_project(project_id: int, push_request: PushRequest):

    task = index_data_content.delay(
        project_id=project_id,
        do_reset=push_request.do_reset,
    )

    return JSONResponse(content={
        "signal": ResponseSignal.DATA_PUSH_QUEUED.value,
        "task_id": task.id,
    })


@nlp_router.get("/index/info/{project_id}")
async def get_project_index_info(request: Request, project_id: int):

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value})

    nlp_controller = build_nlp_controller(request)
    collection_info = await nlp_controller.get_vector_db_collection_info(project=project)

    return JSONResponse(content={
        "signal": ResponseSignal.VECTORDB_COLLECTION_RETRIEVED.value,
        "collection_info": collection_info,
    })


@nlp_router.post("/index/search/{project_id}")
async def search_index(request: Request, project_id: int, search_request: SearchRequest):

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value})

    nlp_controller = build_nlp_controller(request)

    results = await nlp_controller.search_vector_db_collection(
        project=project,
        text=search_request.text,
        limit=search_request.limit,
    )

    if not results:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.VECTORDB_SEARCH_ERROR.value})

    return JSONResponse(content={
        "signal": ResponseSignal.VECTORDB_SEARCH_SUCCESS.value,
        "results": [r.dict() for r in results],
    })


@nlp_router.post("/index/answer/{project_id}")
async def answer_rag(request: Request, project_id: int, search_request: SearchRequest):

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value})

    nlp_controller = build_nlp_controller(request)

    answer, full_prompt, chat_history, grounding = await nlp_controller.answer_rag_question(
        project=project,
        query=search_request.text,
        limit=search_request.limit,
    )

    if not answer:
        record_rag_answer(model="none", outcome="error")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.RAG_ANSWER_ERROR.value})

    return JSONResponse(content={
        "signal": ResponseSignal.RAG_ANSWER_SUCCESS.value,
        "answer": answer,
        "full_prompt": full_prompt,
        "chat_history": chat_history,
        "grounding": grounding,
    })
