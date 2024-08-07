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

def load_model_metadata(file_path):
    model_metadata = {}
    with open(file_path, 'r') as f:
        for line in f:
            model_info = json.loads(line)
            model_metadata[model_info['id']] = model_info
    return model_metadata

def convert_to_json(args):
    if isinstance(args, (dict, list, tuple)):
        return json.dumps(args)
    else:
        return args

def get_model_input_format(model_id):
    model_metadata = load_model_metadata('resources/huggingface-models-metadata.jsonl')
    model_info = model_metadata.get(model_id)
    if not model_info:
        raise ValueError(f"Model {model_id} not found in metadata.")
    
    return model_info.get('inputs', {})

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

HUGGINGFACE_TASKS = {
    "cxr-to-report-generation": CXRToReportGeneration,
    "report-to-cxr-generation": ReportToCXRGeneration,
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
