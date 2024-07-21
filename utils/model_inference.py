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
from utils.resources import (
    audio_from_bytes,
    encode_audio,
    encode_image,
    get_prompt_resource,
    get_resource_url,
    image_from_bytes,
    load_image,
    save_audio,
    save_image,
)
from utils.task_parsing import Task

logger = logging.getLogger(__name__)


# @wrap_exceptions(ModelInferenceException, "Error during model inference")
def infer(task: Task, model_id: str, llm: BaseLLM, session: requests.Session):
    """Execute a task either with LLM or huggingface inference API."""
    if model_id == "openai":
        return infer_openai(task=task, llm=llm)
    else:
        return infer_huggingface(task=task, model_id=model_id, session=session)


def infer_openai(task: Task, llm: BaseLLM):
    logger.info("Starting OpenAI inference")
    prompt_template = load_prompt(
        get_prompt_resource("openai-model-inference-prompt.json")
    )
    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    # Need to replace double quotes with single quotes for correct response generation
    output = llm_chain.predict(
        task=task.json(), task_name=task.task, args=task.args, stop=["<im_end>"]
    )
    result = {"generated text": output}
    logger.debug(f"Inference result: {result}")
    return result


def infer_huggingface(task: Task, model_id: str, session: requests.Session):
    logger.info("Starting huggingface inference")

    with open("resources/huggingface-models-metadata.jsonl") as f:
        for line in f:
            model_data = json.loads(line)
            if model_data["id"] == model_id:
                endpoint = model_data["endpoint"]
                break
    
    huggingface_task = create_huggingface_task(task=task, model_id=model_id)
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


def get_model_input_format(model_id):
    model_metadata = load_model_metadata('resources/huggingface-models-metadata.jsonl')
    model_info = model_metadata.get(model_id)
    if not model_info:
        raise ValueError(f"Model {model_id} not found in metadata.")
    
    return model_info.get('inputs', {})

# NLP Tasks


# deepset/roberta-base-squad2 was removed from huggingface_models-metadata.jsonl because it is currently broken
# Example added to task-planning-examples.json compared to original paper
class QuestionAnswering:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        data = {
            "inputs": {
                "question": self.task.args["question"],
                "context": self.task.args["context"]
                if "context" in self.task.args
                else "",
            }
        }
        return json.dumps(data)

    def parse_response(self, response):
        return response.json()


# Example added to task-planning-examples.json compared to original paper
class SentenceSimilarity:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        data = {
            "inputs": {
                "source_sentence": self.task.args["text1"],
                "sentences": [self.task.args["text2"]],
            }
        }
        # Using string to bypass requests' form encoding
        return json.dumps(data)

    def parse_response(self, response):
        return response.json()


# Example added to task-planning-examples.json compared to original paper
class TextClassification:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        return self.task.args["text"]
        # return {"inputs": self.task.args["text"]}

    def parse_response(self, response):
        return response.json()


class TokenClassification:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        return self.task.args["text"]

    def parse_response(self, response):
        return response.json()


# CV Tasks
class VisualQuestionAnswering:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        img_data = encode_image(self.task.args["image"])
        img_base64 = base64.b64encode(img_data).decode("utf-8")
        data = {
            "inputs": {
                "question": self.task.args["text"],
                "image": img_base64,
            }
        }
        return json.dumps(data)

    def parse_response(self, response):
        return response.json()


class DocumentQuestionAnswering:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        img_data = encode_image(self.task.args["image"])
        img_base64 = base64.b64encode(img_data).decode("utf-8")
        data = {
            "inputs": {
                "question": self.task.args["text"],
                "image": img_base64,
            }
        }
        return json.dumps(data)

    def parse_response(self, response):
        return response.json()


class TextToImage:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        return self.task.args["text"]

    def parse_response(self, response):
        image = image_from_bytes(response.content)
        path = save_image(image)
        return {"generated image": path}


class ImageSegmentation:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        return encode_image(self.task.args["image"])

    def parse_response(self, response):
        image_url = get_resource_url(self.task.args["image"])
        image = load_image(image_url)
        colors = []
        for i in range(len(response.json())):
            colors.append(
                (
                    random.randint(100, 255),
                    random.randint(100, 255),
                    random.randint(100, 255),
                    55,
                )
            )
        predicted_results = []
        for i, pred in enumerate(response.json()):
            mask = pred.pop("mask").encode("utf-8")
            mask = base64.b64decode(mask)
            mask = Image.open(BytesIO(mask), mode="r")
            mask = mask.convert("L")

            layer = Image.new("RGBA", mask.size, colors[i])
            image.paste(layer, (0, 0), mask)
            predicted_results.append(pred)
        path = save_image(image)
        return {
            "generated image with segmentation mask": path,
            "predicted": predicted_results,
        }


# Not yet implemented in huggingface inference API
class ImageToImage:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        img_data = encode_image(self.task.args["image"])
        img_base64 = base64.b64encode(img_data).decode("utf-8")
        data = {
            "inputs": {
                "image": img_base64,
            }
        }
        if "text" in self.task.args:
            data["inputs"]["prompt"] = self.task.args["text"]
        return json.dumps(data)

    def parse_response(self, response):
        image = image_from_bytes(response.content)
        path = save_image(image)
        return {"generated image": path}


class ObjectDetection:
    def __init__(self, task: Task):
        self.task = task

    @property
    def inference_inputs(self):
        return encode_image(self.task.args["image"])

    def parse_response(self, response):
        image_url = get_resource_url(self.task.args["image"])
        image = load_image(image_url)
        draw = ImageDraw.Draw(image)
        labels = list(item["label"] for item in response.json())
        color_map = {}
        for label in labels:
            if label not in color_map:
                color_map[label] = (
                    random.randint(0, 255),
                    random.randint(0, 100),
                    random.randint(0, 255),
                )
        for item in response.json():
            box = item["box"]
            draw.rectangle(
                ((box["xmin"], box["ymin"]), (box["xmax"], box["ymax"])),
                outline=color_map[item["label"]],
                width=2,
            )
            draw.text(
                (box["xmin"] + 5, box["ymin"] - 15),
                item["label"],
                fill=color_map[item["label"]],
            )
        path = save_image(image)
        return {
            "generated image with predicted box": path,
            "predicted": response.json(),
        }


# Example added to task-planning-examples.json compared to original paper
class ImageClassification:
    def __init__(self, task: Task, model_id: str):
        self.task = task
        self.model_id = model_id

    @property
    def inference_inputs(self):
        return encode_image(self.task.args["image"])

    def parse_response(self, response):
        return response.json()


class ImageToText:
    def __init__(self, task: Task, model_id: str):
        self.task = task
        self.model_id = model_id

    @property
    def inference_inputs(self):
        return encode_image(self.task.args["image"])

    def parse_response(self, response):
        return {"generated text": response.json()[0].get("generated_text", "")}


# Audio Tasks
class TextToSpeech:
    def __init__(self, task: Task, model_id: str):
        self.task = task
        self.model_id = model_id

    @property
    def inference_inputs(self):
        return self.task.args["text"]

    def parse_response(self, response):
        audio = audio_from_bytes(response.content)
        path = save_audio(audio)
        return {"generated audio": path}


class AudioToAudio:
    def __init__(self, task: Task, model_id: str):
        self.task = task
        self.model_id = model_id

    @property
    def inference_inputs(self):
        return encode_audio(self.task.args["audio"])

    def parse_response(self, response):
        result = response.json()
        blob = result[0].items()["blob"]
        content = base64.b64decode(blob.encode("utf-8"))
        audio = audio_from_bytes(content)
        path = save_audio(audio)
        return {"generated audio": path}


class AutomaticSpeechRecognition:
    def __init__(self, task: Task, model_id: str):
        self.task = task
        self.model_id = model_id

    @property
    def inference_inputs(self):
        return encode_audio(self.task.args["audio"])

    def parse_response(self, response):
        return response.json()


class AudioClassification:
    def __init__(self, task: Task, model_id: str):
        self.task = task
        self.model_id = model_id

    @property
    def inference_inputs(self):
        return encode_audio(self.task.args["audio"])

    def parse_response(self, response):
        return response.json()


class TextGeneration:
    def __init__(self, task: Task, model_id: str):
        self.task = task
        self.model_id = model_id

    @property
    def inference_inputs(self):
        return self.task.args["text"]

    def parse_response(self, response):
        return response.json()


class CXRToReportGeneration:
    def __init__(self, task: Task, model_id: str):
        self.task = task
        self.model_id = model_id

    @property
    def inference_inputs(self):
        if "text" in self.task.args:
            instruction = self.task.args["text"]
        else:
            instruction = "Generate free-text radiology reports for the entered chest X-ray images."
        img_url = self.task.args["image"]
        print(img_url)
        img_array = self.encode_image_from_url(img_url)
        data = {
            "inputs": {
                "instruction": instruction,
                "input": img_array,
            }
        }
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
    def __init__(self, task: Task, model_id: str):
        self.task = task
        self.model_id = model_id

    @property
    def inference_inputs(self):
        model_input_format = get_model_input_format(self.model_id)
        if isinstance(model_input_format, dict):
            data = {
                "inputs": {
                    "instruction": model_input_format["instruction"],
                    "input": self.task.args["text"]
                }
            }
        elif model_input_format == "text":
            data = {
                "inputs": self.task.args["text"]
            }
        else:
            raise ValueError("Unknown input format for model")

        return data

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
            # response를 직접 처리
            image = Image.open(BytesIO(response))
            
            file_location = self.save_image_s3(image)

            return {"result": {
                "image": file_location,
                "text": ""
            }}
    

# class CXRVQA:
#     def __init__(self, task: Task):
#         self.task = task

#     @property
#     def inference_inputs(self):
#         img_data = encode_image(self.task.args["image"])
#         img_base64 = base64.b64encode(img_data).decode("utf-8")
#         data = {
#             "inputs": {
#                 "instruction": self.task.args["text"],
#                 "input": img_base64,
#             }
#         }
#         return data

#     def save_image_locally(self, image: Image) -> str:
#         buffered = BytesIO()
#         image.save(buffered, format="PNG")
#         buffered.seek(0)
#         upload_file = UploadFile(filename="generated_image.png", file=buffered)
#         file_location = upload_to_s3(upload_file, content_type="image/png")
#         return file_location

#     def parse_response(self, response):
#         rgb_data = response["result_img"]
#         rgb_array = np.array(rgb_data, dtype=np.uint8)
#         image = Image.fromarray(rgb_array, 'RGB')

#         # 로컬에 이미지 저장
#         file_location = self.save_image_locally(image)

#         return {
#             "result": {
#                 "image": file_location,
#                 "text": response["result_text"]
#             }
#         }


HUGGINGFACE_TASKS = {
    "cxr-to-report-generation": CXRToReportGeneration,
    "report-to-cxr-generation": ReportToCXRGeneration,
    # "cxr-visual-qestion-answering": CXRVQA,
}


def create_huggingface_task(task: Task, model_id: str):
    if task.task in HUGGINGFACE_TASKS:
        return HUGGINGFACE_TASKS[task.task](task, model_id)
    else:
        raise NotImplementedError(f"Task {task.task} not supported")


class TaskSummary(BaseModel):
    task: Task
    model_input: Dict[str, str]
    inference_result: Json[Any]
    model: Model
