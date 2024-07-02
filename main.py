from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import logging
import asyncio
import json
from dotenv import load_dotenv
from utils.log import setup_logging
from utils.history import ConversationHistory
from utils.llm_factory import LLMs, create_llms
from utils.task_parsing import parse_tasks, Task
from utils.task_planning import plan_tasks
from utils.model_selection import select_hf_models, Model, Task as ModelTask
from utils.model_inference import infer, TaskSummary
from utils.response_generation import generate_response, format_response


load_dotenv()
# 로깅 설정
setup_logging()
logger = logging.getLogger(__name__)
llms = create_llms()

import sys

# 현재 파이썬 버전을 문자열로 출력
print("Python version:", sys.version)
print("Python executable:", sys.executable)

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

class TaskSummaryResponse(BaseModel):
    task: TaskResponse
    model: str
    inference_result: str

class ModelSelectionRequest(BaseModel):
    user_input: str
    tasks: List[TaskResponse]

class ModelSelectionResponse(BaseModel):
    selected_models: Dict[int, Model]


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
async def select_model(request: ModelSelectionRequest):
    try:
        tasks_sorted = sorted(request.tasks, key=lambda t: max(t.dep))
        # ModelTask 형식으로 변환
        model_tasks = [ModelTask(task=t.task, id=t.id, dep=t.dep, args=t.args) for t in tasks_sorted]
        
        selected_models = await select_hf_models(
            user_input=request.user_input,
            tasks=model_tasks,
            model_selection_llm=llms.model_selection_llm,
            output_fixing_llm=llms.output_fixing_llm,
        )
        return ModelSelectionResponse(selected_models=selected_models)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/execute-tasks", response_model=List[TaskSummaryResponse])
async def execute_tasks(prompt: str = Form(...), history: Optional[str] = Form(None)):
    try:
        history_obj = ConversationHistory.parse_raw(history) if history else ConversationHistory(messages=[])
        
        # Task Planning
        tasks = plan_tasks(prompt, history_obj, llms.task_planning_llm)
        sorted(tasks, key=lambda t: max(t.dep))
        logger.info(f"Sorted tasks: {tasks}")

        # Model Selection
        hf_models = await select_hf_models(
            user_input=prompt,
            tasks=tasks,
            model_selection_llm=llms.model_selection_llm,
            output_fixing_llm=llms.output_fixing_llm,
        )

        # Model Inference
        task_summaries = []
        with requests.Session() as session:
            for task in tasks:
                logger.info(f"Starting task: {task}")
                if task.depends_on_generated_resources():
                    task = task.replace_generated_resources(task_summaries=task_summaries)
                model = hf_models[task.id]
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
                        inference_result=json.dumps(inference_result),
                    )
                )
                logger.info(f"Finished task: {task}")
        logger.info("Finished all tasks")
        logger.debug(f"Task summaries: {task_summaries}")

        return [
            TaskSummaryResponse(
                task=TaskResponse(**summary.task.dict()),
                model=summary.model,
                inference_result=summary.inference_result
            )
            for summary in task_summaries
        ]

    except Exception as e:
        logger.error(f"Failed to execute tasks: {e}")
        return []

