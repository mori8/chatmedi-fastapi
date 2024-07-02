import asyncio
import json
import logging
from collections import defaultdict

from aiohttp import ClientSession

from utils.exceptions import ModelScrapingException, async_wrap_exceptions
from utils.huggingface_api import (HUGGINGFACE_INFERENCE_API_STATUS_URL, get_hf_headers)

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
    task: str, max_description_length: int, top_k: int = 1
):
    """Returns the best k available huggingface models for a given task, sorted by number of likes."""
    # Number of potential candidates changed from top 10 to top_k*2
    # 지금은 태스크별로 후보 하나씩만..
    candidates = HUGGINGFACE_MODELS_MAP[task][0]
    logger.debug(f"Task: {task}; All candidate models: {[c['id'] for c in candidates]}")
    available_models = await filter_available_models(
        candidates=candidates
    )
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
            "tasks": model.get("tasks"),
            "endpoint": model.get("endpoint"),
            "model_card": model.get("model_card", "")[:max_description_length],
        }
        for model in top_k_available_models
    ]
    return top_k_models_info


async def filter_available_models(candidates):
    """Filters out models that are not available or loaded in the huggingface inference API. """
    available_model_ids = [c['id'] for c in candidates if c['endpoint']]
    return [c for c in candidates if c['id'] in available_model_ids]