from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
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
from utils.model_selection import select_hf_model_for_task, Model, Task as ModelTask
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
    inference_result: dict

class GenerateResponseRequest(BaseModel):
    user_input: str
    selected_model: Model
    inference_result: Dict[str, str]

class ModelSelectionRequest(BaseModel):
    user_input: str
    task: str
    context: Any

class ModelSelectionResponse(BaseModel):
    id: str
    reason: str
    task: str
    input_args: Dict[str, str]

class ModelExecutionRequest(BaseModel):
    id: str
    reason: str
    task: str
    input_args: Dict[str, str]

class FinalReportResponse(BaseModel):
    report: str

    


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
async def select_model_endpoint(request: ModelSelectionRequest):
    try:
        selected_model = await select_hf_model_for_task(
            user_input=request.user_input,
            task=request.task,
            context=request.context,
            model_selection_llm=llms.model_selection_llm,
            output_fixing_llm=llms.output_fixing_llm
        )
        return ModelSelectionResponse(
            id=selected_model.id,
            reason=selected_model.reason,
            task=selected_model.task,
            input_args=selected_model.input_args
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


@app.post("/execute-model", response_model=ModelExecutionResult)
async def execute_model(request: ModelExecutionRequest):
    print(request)
    try:
        # Model Inference
        logger.info(f"Starting task: {request.task}")

        with requests.Session() as session:
            inference_result = infer(
                task=request.task,
                model_id=request.id,
                input_args=request.input_args,
                session=session,
            )
        
        logger.info(f"Finished task: {request.task}")
        logger.debug(f"Infernece result: {inference_result['result']}")

        return ModelExecutionResult(
            inference_result=inference_result['result']
        )

    except Exception as e:
        error_message = f"Failed to execute task: {e}\n{traceback.format_exc()}"
        logger.error(error_message)
        return ModelExecutionResult(
            inference_result={"error": error_message}
        )

@app.post("/final-report", response_model=FinalReportResponse)
async def generate_response_endpoint(request: GenerateResponseRequest):
    try:
        report = generate_response(
            # TODO: task_summaries -> execution_results로 변경한거 함수내에서 반영해야함
            user_input=request.user_input,
            selected_model=request.selected_model,
            inference_result=request.inference_result,
            llm=llms.response_generation_llm,  # Ensure llms is defined and properly initialized
        )
        # report = '''### Direct Response\n\n
        # # Based on the provided radiology report, here is the generated chest X-ray image that corresponds to the description of \"Bilateral, diffuse, confluent pulmonary opacities.
        # # Differential diagnoses include severe pulmonary edema ARDS or hemorrhage.\"\n\n![Generated Chest X-ray](https://chatmedi-s3.s3.ap-northeast-2.amazonaws.com/a377d40d-b06f-4e12-8108-8a9bbce35bba.png)\n\n
        # # ### Detailed Workflow\n\n
        # 1. **Task Identification**:\n   - **Task**: Report-to-CXR Generation\n   - **Input**: \"Bilateral, diffuse, confluent pulmonary opacities. Differential diagnoses include severe pulmonary edema ARDS or hemorrhage.\"\n\n
        # 2. **Model Selection**:\n   - **Model Used**: BISPL-KAIST/llm-cxr\n   - **Reason for Selection**: This model is specifically designed for chest X-ray image understanding and generation tasks.
        # It is an instruction-finetuned LLM, which means it can handle complex text inputs like the provided radiology report. The model supports multiple CXR-related tasks, indicating a comprehensive understanding of chest X-ray imagery.\n\n
        # 3. **Inference Process**:\n   - The model processed the input text to generate a corresponding chest X-ray image.\n   - The generated image reflects the described conditions: bilateral, diffuse, confluent pulmonary opacities, which are indicative of severe pulmonary edema, ARDS, or hemorrhage.\n\n
        # 4. **Inference Result**:\n   - **Generated Image**: The image link provided above.\n   - **Text**: No additional text was generated as the primary output was the image.\n\nI hope this meets your needs! If you have any further questions or need additional modifications, feel free to ask.'''
        
        return { "report": report }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/available-models")
def get_available_models(task: str) -> List[str]:
    available_models = [model["id"] for model in alternative_models if task in model["tasks"]]
    if not available_models:
        raise HTTPException(status_code=404, detail="No models available for the given task.")
    return available_models
