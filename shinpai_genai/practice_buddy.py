'''
This contains chains that converse with the student in Japanese.
'''
import os
import json

from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptValue
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chains.conversation.memory import (
    ConversationBufferMemory
)
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory


with open("./config.json", "r") as f:
    CONFIG = json.load(f)

with open("./shinpai_genai/prompts/practice_buddy_system_prompt.txt", "r") as f:
    PRACTICE_BUDDY_SYSTEM_PROMPT = f.read()

def get_practice_chain():
    '''
    Constructs and returns an instance of the practice buddy chain
    '''

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key = CONFIG["openai_api_key"],
        temperature = 0,
    )

    history = ChatMessageHistory()
    history.add_message(SystemMessage(content = PRACTICE_BUDDY_SYSTEM_PROMPT))

    memory = ConversationBufferMemory(
        chat_memory = history,
        human_prefix = "Student",
        ai_prefix = "Practice Buddy",
    )

    practice_chain = ConversationChain(
        llm = llm,
        memory = memory,
    )

    return practice_chain


