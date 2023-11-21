'''
Contains the Language Tutorial Chain.
This consists of both the practice buddy and the critic.
'''

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Extra

from langchain.schema.language_model import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.prompts.base import BasePromptTemplate
from langchain.chains import ConversationChain, LLMChain

from shinpai_genai.practice_buddy import get_practice_chain
from shinpai_genai.critic.simple_critic import get_critic_chain
from shinpai_genai.critic.referential_critic import ReferentialCriticChain
from shinpai_genai.critic.utils import get_conversation_string

class LanguageTutorialChain(Chain):
    '''
    Will converse with the user in Japanese.
    And in a separate stream, also provides feedback from a critic.
    '''

    practice_buddy_chain: ConversationChain = get_practice_chain()
    critic_chain: LLMChain = ReferentialCriticChain()

    @property
    def input_keys(self) -> List[str]:
        '''
        Input is just the latest query of the user/student
        '''
        return ["query"]

    @property
    def output_keys(self) -> List[str]:

        return ["response", "feedback"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        # prompt_value = self.prompt.format_prompt(**inputs)

        practice_response = self.practice_buddy_chain(
            inputs["query"],
            callbacks = run_manager.get_child() if run_manager else None,
        )
        conversation_str = get_conversation_string(self.practice_buddy_chain)

        critic_response = self.critic_chain.run(
            {"conversation": conversation_str},
            callbacks = run_manager.get_child() if run_manager else None,
        )


        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        # response = self.llm.generate_prompt(
        #     [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        # )

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            run_manager.on_text("Log something about this run")

        return {
            "response": practice_response["response"],
            "feedback": critic_response,
        }