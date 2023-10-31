'''
A critic that refers to curriculum and grammar material when giving feedback.
'''
from __future__ import annotations

import os
import json

from typing import Any, Dict, List, Optional

from langchain.schema.language_model import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)

from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    BasePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptValue
)
from langchain.chains import LLMChain
from shinpai_genai.critic.utils import get_conversation_string

with open("./config.json", "r") as f:
    CONFIG = json.load(f)

with open("./shinpai_genai/prompts/lesson_selector_system_prompt.txt", "r") as f:
    LESSON_SELECTOR_SYSTEM_PROMPT = f.read()

with open("./shinpai_genai/prompts/lesson_selector_user_prompt.txt", "r") as f:
    LESSON_SELECTOR_USER_PROMPT = f.read()

with open("./data/grammar_documents/n4_grammar/index_short.json", "r", encoding='utf-8') as f:
    # NOTE: read as string
    LESSON_LIST = f.read().encode('utf-8').decode('unicode_escape')

# NOTE: can't use curly braces with prompt template
    # because it confuses it for placeholders
LESSON_LIST = LESSON_LIST.replace("{", "(").replace("}", ")")

def get_lesson_selector_chain(**kwargs):
    '''
    Constructs a chain which can select lessons for giving feedback to the student.
    '''

    lesson_list = kwargs.get("lesson_list", LESSON_LIST)

    llm = kwargs.get(
        "llm", 
        ChatOpenAI(
            model_name="gpt-3.5-turbo-16k",
            openai_api_key = CONFIG["openai_api_key"],
            temperature = 0,
        )
    )

    system_prompt = LESSON_SELECTOR_SYSTEM_PROMPT.format(lesson_list = lesson_list)

    lesson_selector_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", LESSON_SELECTOR_USER_PROMPT),
    ])

    lesson_selector_chain = LLMChain(
        llm = llm,
        prompt = lesson_selector_prompt
    )

    return lesson_selector_chain

class ReferentialCriticChain(Chain):
    '''
    Provides feedback for the student during conversation.
    Refers to lessons when giving feedback.

    TODO: If something was wrong in the grammar, prioritize correcting that before using other lessons
        See if we can do this using the prompt, or have to chain it.
    '''

    pass