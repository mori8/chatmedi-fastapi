import base64
import json
import logging
import random
import numpy as np
import base64
import boto3
from uuid import uuid4
from io import BytesIO
from typing import Any, Dict
from pathlib import Path

import requests
from PIL import Image, ImageDraw
from langchain.chains import LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts import load_prompt
from pydantic import BaseModel, Json
from fastapi import FastAPI, File, UploadFile

from utils.exceptions import ModelInferenceException, wrap_exceptions
from utils.save_file import upload_to_s3
from utils.huggingface_api import (HUGGINGFACE_INFERENCE_API_URL, get_hf_headers)
from utils.model_selection import Model
from utils.task_parsing import Task

logger = logging.getLogger(__name__)


# @wrap_exceptions(ModelInferenceException, "Error during model inference")
def infer(task: str, model_id: str, input_args: Any, session: requests.Session):
    return infer_huggingface(task=task, model_id=model_id, input_args=input_args, session=session)

def infer_huggingface(task: str, model_id: str, input_args: Any, session: requests.Session):
    logger.info("Starting huggingface inference")

    with open("resources/huggingface-models-metadata.jsonl") as f:
        for line in f:
            model_data = json.loads(line)
            if model_data["id"] == model_id:
                endpoint = model_data["endpoint"]
                break
    
    huggingface_task = create_huggingface_task(task=task, model_id=model_id, input_args=input_args)
    data = huggingface_task.inference_inputs
    # print(data)
    headers = get_hf_headers()
    # print(headers)
    try:
        response = session.post(endpoint, headers=headers, json=data)
        response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
        result = huggingface_task.parse_response(response.json())
        logger.debug(f"Inference result: {result}")
        return result
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error occurred: {e}")
        logger.error(f"Response status code: {e.response.status_code}")
        logger.error(f"Response headers: {e.response.headers}")
        logger.error(f"Response body: {e.response.text}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        raise

# NLP Tasks

class CXRToReportGeneration:
    def __init__(self, task: str, model_id: str, input_args: Any):
        self.task = task
        self.model_id = model_id
        self.input_args = input_args

    @property
    def inference_inputs(self):
        data = self.convert_image_urls(self.input_args)
        return {"inputs": data}

    def convert_image_urls(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                data[key] = self.convert_image_urls(value)
        elif isinstance(data, list):
            data = [self.convert_image_urls(item) for item in data]
        elif isinstance(data, str) and data.startswith("https://chatmedi-s3.s3.ap-northeast-2.amazonaws.com"):
            data = self.encode_image_from_url(data)

        return data

    def encode_image_from_url(self, url):
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img = img.convert("RGB")
        img_array = np.array(img)
        return img_array.tolist()

    def parse_response(self, response):
        return {
            "result": {
                "image": "",
                "text": response["result_text"]
            }
        }


class ReportToCXRGeneration:
    def __init__(self, task: str, model_id: str, input_args: Any):
        self.task = task
        self.model_id = model_id
        self.input_args = input_args

    @property
    def inference_inputs(self):
        return {"inputs": self.input_args}

    def save_image_s3(self, image: Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        upload_file = UploadFile(filename="generated_image.png", file=buffered)
        file_location = upload_to_s3(upload_file, content_type="image/png")
        return file_location

    def parse_response(self, response):
        if "result_img" in response:
            rgb_data = response["result_img"]
            rgb_array = np.array(rgb_data, dtype=np.uint8)
            image = Image.fromarray(rgb_array, 'RGB')
            
            file_location = self.save_image_s3(image)

            return {"result": {
                "image": file_location,
                "text": ""
            }}
        else:
            image_data = base64.b64decode(response)
            image = Image.open(BytesIO(image_data))

            file_location = self.save_image_s3(image)

            return {"result": {
                "image": file_location,
                "text": ""
            }}


class ClinicalNoteAnalysis:
    def __init__(self, task: str, model_id: str, input_args: Any):
        self.task = task
        self.model_id = model_id
        self.input_args = input_args
        self.prompt = """You are an intelligent clinical language model.
        Below is a snippet of patient's discharge summary and a following instruction from healthcare professional.
        Write a response that appropriately completes the instruction.
        The response should provide the accurate answer to the instruction, while being concise.

        [Discharge Summary Begin]
        {note}
        [Discharge Summary End]

        [Instruction Begin]
        {question}
        [Instruction End] 
        """

    @property
    def inference_inputs(self):
        note = self.input_args.get("note", "")
        question = self.input_args.get("question", "")
        model_input = self.prompt.format(note=note, question=question)
        return {"inputs": model_input}

    def parse_response(self, response):
        return {
            "result": response
        }

HUGGINGFACE_TASKS = {
    "cxr-to-report-generation": CXRToReportGeneration,
    "report-to-cxr-generation": ReportToCXRGeneration,
    "clinical-note-analysis": ClinicalNoteAnalysis
}

def create_huggingface_task(task: str, model_id: str, input_args: Any):
    if task in HUGGINGFACE_TASKS:
        return HUGGINGFACE_TASKS[task](task, model_id, input_args)
    else:
        raise NotImplementedError(f"Task {task} not supported")


class TaskSummary(BaseModel):
    task: Task
    model_input: Dict[str, str]
    inference_result: Json[Any]
    model: Model
