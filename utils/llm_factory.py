import logging
from collections import namedtuple

import tiktoken
from langchain_openai import ChatOpenAI as OpenAI
from langchain_anthropic import ChatAnthropic as Anthropic

LLM_NAME = "gpt-4o"
# Encoding for text-davinci-003
ENCODING_NAME = "cl100k_base"
ENCODING = tiktoken.get_encoding(ENCODING_NAME)
# Max input tokens for gpt-3.5-turbo
LLM_MAX_TOKENS = 4096

# As specified in huggingGPT paper
TASK_PLANNING_LOGIT_BIAS = 0.1
MODEL_SELECTION_LOGIT_BIAS = 5

logger = logging.getLogger(__name__)

LLMs = namedtuple(
    "LLMs",
    [
        "task_planning_llm",
        "model_selection_llm",
        "model_inference_llm",
        "response_generation_llm",
        "output_fixing_llm",
    ],
)


def create_llms() -> LLMs:
    """Create various LLM agents according to the huggingGPT paper's specifications."""
    logger.info(f"Creating {LLM_NAME} LLMs")

    task_parsing_highlight_ids = get_token_ids_for_task_parsing()
    choose_model_highlight_ids = get_token_ids_for_choose_model()

    task_planning_llm = OpenAI(
        model_name=LLM_NAME,
        temperature=0,
        logit_bias={
            token_id: TASK_PLANNING_LOGIT_BIAS
            for token_id in task_parsing_highlight_ids
        },
    )
    model_selection_llm = Anthropic(
        model_name="claude-3-5-sonnet-20240620",
        temperature=0
    )
    model_inference_llm = OpenAI(model_name=LLM_NAME, temperature=0)
    response_generation_llm = OpenAI(model_name=LLM_NAME, temperature=0)
    output_fixing_llm = OpenAI(model_name=LLM_NAME, temperature=0)
    return LLMs(
        task_planning_llm=task_planning_llm,
        model_selection_llm=model_selection_llm,
        model_inference_llm=model_inference_llm,
        response_generation_llm=response_generation_llm,
        output_fixing_llm=output_fixing_llm,
    )


def get_token_ids_for_task_parsing() -> list[int]:
    text = """{"task": "question-answering-about-medical-domain", "cxr-image-manipulation", "endoscope-image-manipulation", "text-to-molecule", "cxr-report-generation", "molecule-to-text", "ct-image-manipulation", "ct-reconstruction", "args", "text", "path", "dep", "id", "<GENERATED>-"}"""
    res = ENCODING.encode(text)
    res = list(set(res))
    return res


def get_token_ids_for_choose_model() -> list[int]:
    text = """{"id": "reason"}"""
    res = ENCODING.encode(text)
    res = list(set(res))
    return res


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))
