import asyncio
import json
import logging
from collections import defaultdict

from aiohttp import ClientSession

from utils.exceptions import ModelScrapingException, async_wrap_exceptions

logger = logging.getLogger(__name__)

def read_huggingface_models_metadata():
    """Reads the metadata of all huggingface models from the local models cache file."""
    with open("resources/huggingface-models-metadata.jsonl") as f:
        models = [json.loads(line) for line in f]
    models_map = defaultdict(list)
    for model in models:
        for task in model["tasks"]:
            models_map[task].append(model)
    return models_map

HUGGINGFACE_MODELS_MAP = read_huggingface_models_metadata()

@async_wrap_exceptions(
    ModelScrapingException,
    "Failed to find compatible models already loaded in the huggingface inference API.",
)
async def get_top_k_models(
    task: str, top_k: int, max_description_length: int, session: ClientSession
):
    """Returns the best k available huggingface models for a given task, sorted by number of likes."""
    # Number of potential candidates changed from top 10 to top_k*2
    candidates = HUGGINGFACE_MODELS_MAP[task][: top_k * 2]
    logger.debug(f"Task: {task}; All candidate models: {[c['id'] for c in candidates]}")
    available_models = filter_available_models(candidates)
    logger.debug(
        f"Task: {task}; Available models: {[c['id'] for c in available_models]}"
    )
    top_k_available_models = available_models[:top_k]
    if not top_k_available_models:
        raise Exception(f"No available models for task: {task}")
    logger.debug(
        f"Task: {task}; Top {top_k} available models: {[c['id'] for c in top_k_available_models]}"
    )
    top_k_models_info = [
        {
            "id": model["id"],
            "model_card": model.get("model_card", "")[:max_description_length],
            "tasks": model.get("tasks", []),
            "endpoint": model.get("endpoint", ""),
        }
        for model in top_k_available_models
    ]
    return top_k_models_info

def filter_available_models(candidates):
    """Filters out models that do not have an endpoint defined in the metadata."""
    available_models = [c for c in candidates if c.get("endpoint")]
    return available_models

def load_model_metadata(file_path):
    model_metadata = {}
    with open(file_path, 'r') as f:
        for line in f:
            model_info = json.loads(line)
            model_metadata[model_info['id']] = model_info
    return model_metadata

def get_single_model_info(model_id):
    model_metadata = load_model_metadata('resources/huggingface-models-metadata.jsonl')
    model_info = model_metadata.get(model_id)
    if not model_info:
        raise ValueError(f"Model {model_id} not found in metadata.")
    return model_info