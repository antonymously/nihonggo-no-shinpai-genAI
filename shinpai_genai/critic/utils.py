
def get_conversation_string(conversational_chain, **kwargs):
    '''
    Extracts the conversation history as a string.
    Used to put the conversation history in the prompt of another chain.
    '''

    messages = conversational_chain.memory.dict()["chat_memory"]["messages"][:-1]
        # NOTE:
            # exclude the last one because it will end with the ai message
            # want to get the last human message

    conversation_str = get_conversation_string_from_messages(messages, **kwargs)
    
    return conversation_str

def get_conversation_string_from_messages(messages, **kwargs):

    human_label = kwargs.get("human_label", "STUDENT")
    ai_label = kwargs.get("ai_label", "PRACTICE BUDDY")

    conversation_str = ""

    for msg_dict in messages:
        
        if msg_dict["type"] == "system":
            continue
        elif msg_dict["type"] == 'human':
            conversation_str += human_label + ": " + msg_dict["content"] + "\n"
        elif msg_dict["type"] == 'ai':
            conversation_str += ai_label + ": " + msg_dict["content"] + "\n"
    
    return conversation_str