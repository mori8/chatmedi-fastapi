from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import logging
import traceback
import asyncio
import requests
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv

from utils.save_file import upload_to_s3
from utils.log import setup_logging
from utils.history import ConversationHistory
from utils.llm_factory import LLMs, create_llms
from utils.task_parsing import parse_tasks, Task
from utils.task_planning import plan_tasks
from utils.model_selection import select_hf_models, Model, Task as ModelTask
from utils.model_inference import infer, TaskSummary
from utils.response_generation import generate_response


load_dotenv()
# 로깅 설정
setup_logging()
logger = logging.getLogger(__name__)
llms = create_llms()


app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TaskResponse(BaseModel):
    task: str
    id: int
    dep: List[int]
    args: Dict[str, str]

class ModelExecutionResult(BaseModel):
    inference_result: Dict[str, str]

class GenerateResponseRequest(BaseModel):
    user_input: str
    execution_result: ModelExecutionResult

class ModelSelectionRequest(BaseModel):
    user_input: str
    tasks: List[TaskResponse]

class ModelSelectionResponse(BaseModel):
    prompt: str
    selected_models: Dict[int, Model]

class ModelExecutionRequest(BaseModel):
    user_input: str
    selected_model: Model


with open("resources/huggingface-models-metadata.jsonl") as f:
    alternative_models = [json.loads(line) for line in f]


@app.post("/parse-task", response_model=List[TaskResponse])
async def parse_task(prompt: str = Form(...), image: Optional[UploadFile] = File(None)):
    # task_parsing 모듈의 기능을 사용하여 프롬프트를 파싱
    try:
        tasks = parse_tasks(prompt)
        return [TaskResponse(**task.dict()) for task in tasks]
    except Exception as e:
        logger.error(f"Failed to parse tasks: {e}")
        return []

@app.post("/plan-task", response_model=List[TaskResponse])
async def plan_task(prompt: str = Form(...), history: Optional[str] = Form(None)):
    # history를 ConversationHistory 객체로 변환 (가정: JSON 문자열 형태로 전달됨)
    try:
        history_obj = ConversationHistory.parse_raw(history) if history else ConversationHistory(messages=[])
        tasks = plan_tasks(prompt, history_obj, llms.task_planning_llm)
        return [TaskResponse(**task.dict()) for task in tasks]
    except Exception as e:
        logger.error(f"Failed to plan tasks: {e}")
        return []
    
@app.post("/select-model", response_model=ModelSelectionResponse)
def select_model(request: ModelSelectionRequest):
    response = {
        "id": "llm-dxr",
        "reason": "This model is the best model for the given task.",
        "task": "cxr-to-report-generation",
        "input_args": {
            "instruction": "Generate a report based on the given chest X-ray image.",
            "input": "[]",
        },
    }
    return response

@app.post("/execute-tasks", response_model=List[ModelExecutionResult])
async def execute_tasks(request: ModelExecutionRequest):
    try:
        # Model Inference
        task_summaries = []
        with requests.Session() as session:
            for task in request.tasks:
                logger.info(f"Starting task: {task}")
                if task.depends_on_generated_resources():
                    task = task.replace_generated_resources(task_summaries=task_summaries)
                model = request.selected_models[task.id]
                inference_result = infer(
                    task=task,
                    model_id=model.id,
                    llm=llms.model_inference_llm,
                    session=session,
                )
                task_summaries.append(
                    TaskSummary(
                        task=task,
                        model=model,
                        model_input=task.args,
                        inference_result=json.dumps(inference_result),
                    )
                )
                logger.info(f"Finished task: {task}")
        logger.info("Finished all tasks")
        logger.debug(f"Task summaries: {task_summaries}")

        return [
            ModelExecutionResult(
                task=Task(**summary.task.dict()),  # TaskResponse를 Task로 변환
                model=summary.model,
                model_input=summary.model_input,
                inference_result=summary.inference_result["result"]
            )
            for summary in task_summaries
        ]

    except Exception as e:
        error_message = f"Failed to execute tasks: {e}\n{traceback.format_exc()}"
        logger.error(error_message)
        return []
    
@app.post("/execute-model", response_model=ModelExecutionResult)
async def execute_model(request: ModelExecutionRequest):
    try:
        # Model Inference
        # task = request.selected_model.task  # 단일 task 가정
        # logger.info(f"Starting task: {task}")
        
        # if task.depends_on_generated_resources():
        #     task = task.replace_generated_resources(task_summaries=[])

        # model = request.selected_models[task.id]
        
        # with requests.Session() as session:
        #     # TODO: infer에서 task가 string으로 입력되는거 반영해야함
        #     inference_result = infer(
        #         task=task,
        #         model_id=model.id,
        #         llm=llms.model_inference_llm,
        #         session=session,
        #     )
        
        # task_summary = TaskSummary(
        #     task=task,
        #     model=model,
        #     model_input=task.args,
        #     inference_result=json.dumps(inference_result),
        # )
        
        # logger.info(f"Finished task: {task}")
        # logger.debug(f"Task summary: {task_summary}")

        # return ModelExecutionResult(
        #     task=task_summary.task,  # TaskResponse를 Task로 변환
        #     model=task_summary.model,
        #     inference_result=task_summary.inference_result["result"]
        # )
        return ModelExecutionResult(
            inference_result={
                "result": "This is the result of the model execution."
            }
        )

    except Exception as e:
        error_message = f"Failed to execute task: {e}\n{traceback.format_exc()}"
        logger.error(error_message)
        return ModelExecutionResult(
            inference_result=None
        )

@app.post("/generate-response")
async def generate_response_endpoint(request: GenerateResponseRequest):
    try:
        response = generate_response(
            # TODO: task_summaries -> execution_results로 변경한거 함수내에서 반영해야함
            user_input=request.user_input,
            execution_result=request.execution_result,
            llm=llms.response_generation_llm,  # Ensure llms is defined and properly initialized
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    file_location = upload_to_s3(file)
    return {"filename": file.filename, "url": f"/files/{file.filename}"}


@app.get("/available-models")
def get_available_models(task: str) -> List[str]:
    available_models = [model["id"] for model in alternative_models if task in model["tasks"]]
    if not available_models:
        raise HTTPException(status_code=404, detail="No models available for the given task.")
    return available_models

# @app.get("/files/{filename}")
# async def get_image(filename: str):
#     file_location = UPLOAD_DIR / filename
#     if file_location.exists():
#         return FileResponse(file_location)
#     return {"error": "File not found"}