'''
A critic that refers to curriculum and grammar material when giving feedback.
'''
from __future__ import annotations

import os
import json
from ast import literal_eval

from typing import Any, Dict, List, Optional

from copy import copy
import numpy as np

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
LESSON_LIST_SAFE = LESSON_LIST.replace("{", "(").replace("}", ")")

# for the variety lesson selector
    # which shuffles the lessons
LESSON_LIST_DICTS = json.loads(LESSON_LIST)

def get_lesson_selector_chain(**kwargs):
    '''
    Constructs a chain which can select lessons for giving feedback to the student.
    '''

    lesson_list = kwargs.get("lesson_list", LESSON_LIST_SAFE)
    n_llm_lessons = kwargs.get("n_llm_lessons", 3)

    llm = kwargs.get(
        "llm", 
        ChatOpenAI(
            model_name="gpt-3.5-turbo-16k",
            openai_api_key = CONFIG["openai_api_key"],
            temperature = 0,
        )
    )

    system_prompt = LESSON_SELECTOR_SYSTEM_PROMPT.format(
        lesson_list = lesson_list,
        n_llm_lessons = n_llm_lessons,
    )

    lesson_selector_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", LESSON_SELECTOR_USER_PROMPT),
    ])

    lesson_selector_chain = LLMChain(
        llm = llm,
        prompt = lesson_selector_prompt
    )

    return lesson_selector_chain

class VarietyLessonSelectorChain(Chain):
    '''
    Tries to introduce more variety to lesson selection.
        Shuffles lesson list everytime before prompting to distribute bias.
        Takes some lessons from the selection of the LLM, but adds some random ones.
    '''

    n_llm_lessons: int = 2
    n_random_lessons: int = 1

    random_state: np.random.RandomState = np.random.RandomState(seed = 2023)

    # NOTE: user is not expected to change this chain
    llm_chain: LLMChain = LLMChain(
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-16k",
            openai_api_key = CONFIG["openai_api_key"],
            temperature = 0,
        ),
        prompt = ChatPromptTemplate.from_messages([
            ("system", LESSON_SELECTOR_SYSTEM_PROMPT),
            ("human", LESSON_SELECTOR_USER_PROMPT),
        ]),
    )

    @property
    def input_keys(self) -> List[str]:

        return ["lesson_list", "conversation"]

    @property
    def output_keys(self) -> List[str]:

        return ["lesson_ids"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:

        lesson_list_dicts = copy(inputs["lesson_list"])

        # shuffle the lesson list
            # NOTE: inplace
        self.random_state.shuffle(lesson_list_dicts)

        # make the lesson list into a string
            # ensure the encoding is correct
        lesson_list_str = json.dumps(lesson_list_dicts, ensure_ascii=False)

        # run the llm chain with the lesson list string and conversation string as input
        llm_lessons_str = self.llm_chain.run(
            {
                "conversation": inputs["conversation"],
                "lesson_list": lesson_list_str,
                "n_llm_lessons": self.n_llm_lessons,
            }
        )

        # convert the output to list
        llm_lessons_list = literal_eval(llm_lessons_str)

        # get a list of indices of lessons
            # exclude the indices selected by the llm
        lesson_ids = [l["id"] for l in lesson_list_dicts if l["id"] not in llm_lessons_list]

        # select random lessons from the remaining
            # NOTE: outputs a list even if size = 1
        rand_lessons = self.random_state.choice(lesson_ids, size = self.n_random_lessons, replace=False).tolist()

        # concatenate the two lists
        chosen_ids = llm_lessons_list + rand_lessons

        # log on run manager to ensure that things went right
        if run_manager:
            run_manager.on_text("Total number of lessons: {}".format(len(lesson_list_dicts)))
            run_manager.on_text("Lessons selected by llm: {}".format(llm_lessons_list))
            run_manager.on_text("Remaining after llm: {}".format(len(lesson_ids)))
            run_manager.on_text("Selected by random: {}".format(rand_lessons))

        return {
            "lesson_ids": chosen_ids,
        }

class ReferentialCriticChain(Chain):
    '''
    Provides feedback for the student during conversation.
    Refers to lessons when giving feedback.

    TODO: If something was wrong in the grammar, prioritize correcting that before using other lessons
        See if we can do this using the prompt, or have to chain it.

    TODO: Probably should have some visibility on previous feedback.
        So it can comment on whether the student applied it correctly.
        Add this to prompt?
    '''

    pass