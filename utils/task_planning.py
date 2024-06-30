import logging

from langchain import LLMChain
from langchain.llms.base import BaseLLM
from langchain.chat_models import ChatOpenAI as OpenAI
from langchain.prompts import load_prompt
from langchain.prompts import PromptTemplate

from utils.exceptions import TaskPlanningException, wrap_exceptions
from utils.history import ConversationHistory
from utils.llm_factory import LLM_MAX_TOKENS, count_tokens
from utils.resources import get_prompt_resource
from utils.task_parsing import Task, parse_tasks

logger = logging.getLogger(__name__)

MAIN_PROMPT_TOKENS = 800
MAX_HISTORY_TOKENS = LLM_MAX_TOKENS - MAIN_PROMPT_TOKENS


@wrap_exceptions(TaskPlanningException, "Failed to plan tasks")
def plan_tasks(
    user_input: str, history: ConversationHistory, llm: BaseLLM
) -> list[Task]:
    """Use LLM agent to plan tasks in order solve user request."""
    logger.info("Starting task planning")
    task_planning_prompt_template = load_prompt(
        get_prompt_resource("task-planning-few-shot-prompt.json")
    )
    llm_chain = LLMChain(prompt=task_planning_prompt_template, llm=llm)
    history_truncated = truncate_history(history)
    logger.info(f"user input: {user_input}")
    output = llm_chain.predict(
        user_input=user_input, history=history_truncated,
        stop=["<im_end>"]
    )

    logger.info(f"Task planning raw output: {output}")
    tasks = parse_tasks(output)
    return tasks


def truncate_history(history: ConversationHistory) -> ConversationHistory:
    """Truncate history to fit within the max token limit for the task planning LLM"""
    example_prompt_template = load_prompt(
        get_prompt_resource("task-planning-example-prompt.json")
    )
    token_counter = 0
    n_messages = 0
    # Iterate through history backwards in pairs, to ensure most recent messages are kept
    for i in range(0, len(history), 2):
        user_message = history[-(i + 2)]
        assistant_message = history[-(i + 1)]
        # Turn messages into LLM prompt string
        history_text = example_prompt_template.format(
            example_input=user_message["content"],
            example_output=assistant_message["content"],
        )
        n_tokens = count_tokens(history_text)
        if token_counter + n_tokens <= MAX_HISTORY_TOKENS:
            n_messages += 2
            token_counter += n_tokens
        else:
            break
    start = len(history) - n_messages
    return history[start:]
