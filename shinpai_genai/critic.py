'''
This contains chains which provide feedback for the student during the conversation.
'''

import os
import json

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptValue
)
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory

with open("./config.json", "r") as f:
    CONFIG = json.load(f)

with open("./shinpai_genai/prompts/critic_system_prompt.txt", "r") as f:
    CRITIC_SYSTEM_PROMPT = f.read()

with open("./shinpai_genai/prompts/critic_user_prompt_template.txt", "r") as f:
    CRITIC_USER_PROMPT_TEMPLATE = f.read()

def get_conversation_string(conversational_chain, **kwargs):
    '''
    Extracts the conversation history as a string.
    Used to put the conversation history in the prompt of another chain.
    '''
    human_label = kwargs.get("human_label", "STUDENT")
    ai_label = kwargs.get("ai_label", "PRACTICE BUDDY")

    conversation_str = ""

    for msg_dict in conversational_chain.memory.dict()["chat_memory"]["messages"][:-1]:
        # NOTE:
            # exclude the last one because it will end with the ai message
            # want to get the last human message

        if msg_dict["type"] == "system":
            continue
        elif msg_dict["type"] == 'human':
            conversation_str += human_label + ": " + msg_dict["content"] + "\n"
        elif msg_dict["type"] == 'ai':
            conversation_str += ai_label + ": " + msg_dict["content"] + "\n"
    
    return conversation_str

def get_critic_chain():
    '''
    Constructs and returns an instance of the critic chain
    '''

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key = CONFIG["openai_api_key"],
        temperature = 0,
    )

    # critic has no chat history
        # just pass the prompt and new conversation everytime

    critic_prompt = ChatPromptTemplate.from_messages([
        ("system", CRITIC_SYSTEM_PROMPT),
        ("human", CRITIC_USER_PROMPT_TEMPLATE),
    ])

    critic_chain = LLMChain(
        llm = llm,
        prompt = critic_prompt
    )

    return critic_chain



