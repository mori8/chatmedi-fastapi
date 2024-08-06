import asyncio
import json
import logging

import aiohttp
from langchain.chains import LLMChain
from langchain.llms.base import BaseLLM
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import load_prompt
from pydantic import BaseModel, Field
from typing import Any

from utils.exceptions import ModelSelectionException, async_wrap_exceptions
from utils.model_scraper import get_top_k_models
from utils.resources import get_prompt_resource
from utils.task_parsing import Task

logger = logging.getLogger(__name__)


class Model(BaseModel):
    id: str = Field(description="ID of the model")
    reason: str = Field(description="Reason for selecting this model")
    task: str
    input_args: Any


# @async_wrap_exceptions(ModelSelectionException, "Failed to select model")
async def select_model(
    task: str,
    context: Any,
    user_input: str,
    model_selection_llm: BaseLLM,
    output_fixing_llm: BaseLLM,
):
    logger.info(f"Starting model selection for task: {task}")

    async with aiohttp.ClientSession() as session:
        try:
            top_k_models = await get_top_k_models(
                task=task, top_k=5, max_description_length=100, session=session
            )
        except Exception as e:
            logger.error(f"Error fetching top k models for task {task}: {e}")
            raise

    try:
        prompt_template = load_prompt(
            get_prompt_resource("model-selection-prompt.json")
        )
        llm_chain = LLMChain(prompt=prompt_template, llm=model_selection_llm)

        # Prepare the context string
        context_str = ""
        if hasattr(context, 'text') and context.text:
            context_str += f"Text: {context.text} "
        if hasattr(context, 'image') and context.image:
            context_str += f"Image: {context.image}"

        context_str = context_str.strip()  # Remove any trailing whitespace

        models_str = json.dumps(top_k_models).replace('"', "'")
        output = await llm_chain.apredict(
            user_input=user_input, task=task, context=context_str, models=models_str, stop=["<im_end>"]
        )
        logger.debug(f"Model selection raw output: {output}")

        parser = PydanticOutputParser(pydantic_object=Model)
        fixing_parser = OutputFixingParser.from_llm(
            parser=parser, llm=output_fixing_llm
        )
        model = fixing_parser.parse(output)
    except Exception as e:
        logger.error(f"Error during model selection for task {task}: {e}")
        raise

    logger.info(f"For task: {task}, selected model: {model}")
    return model


async def select_hf_model_for_task(
    user_input: str,
    task: str,
    context: Any,
    model_selection_llm: BaseLLM,
    output_fixing_llm: BaseLLM,
) -> Model:
    """Use LLM agent to select the best available HuggingFace model for a given task, given model metadata."""
    try:
        model = await select_model(
            user_input=user_input,
            task=task,
            context=context,
            model_selection_llm=model_selection_llm,
            output_fixing_llm=output_fixing_llm,
        )
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise
    return model