import json
import logging

from langchain.chains import LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts import load_prompt
from pydantic import BaseModel
from typing import Dict, List

from utils.task_parsing import Task
from utils.model_selection import Model
from utils.exceptions import ResponseGenerationException, wrap_exceptions
from utils.resources import get_prompt_resource, prepend_resource_dir

logger = logging.getLogger(__name__)

class TaskSummariesForFinalResponse(BaseModel):
    task: Task
    model: Model
    inference_result: Dict[str, str]


# @wrap_exceptions(ResponseGenerationException, "Failed to generate assistant response")
def generate_response(
    user_input: str, selected_model: Model, inference_result: Dict[str, str], llm: BaseLLM
) -> str:
    """Use LLM agent to generate a response to the user's input, given task results."""
    logger.info("Starting response generation")
    prompt_template = load_prompt(
        get_prompt_resource("response-generation-prompt.json")
    )
    infefence_result_str = json.dumps(inference_result)
    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    response = llm_chain.predict(
        user_input=user_input, selected_model=selected_model, inference_results=infefence_result_str, stop=["<im_end>"]
    )
    logger.info(f"Generated response: {response}")
    return response


def format_response(response: str) -> str:
    """Format the response to be more readable for user."""
    response = response.strip()
    response = prepend_resource_dir(response)
    return response


def task_summaries_to_json(task_summaries: List[TaskSummariesForFinalResponse]) -> str:
    dicts = [ts.dict() for ts in task_summaries]
    return json.dumps(dicts)