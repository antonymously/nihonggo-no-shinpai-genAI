'''
Runs shinpai-genai on terminal.
'''

from shinpai_genai.practice_buddy import get_practice_chain
from shinpai_genai.critic import get_critic_chain, get_conversation_string


def main():

    practice_chain = get_practice_chain()
    critic_chain = get_critic_chain()

    print("PRACTICE BUDDY: 何か教えてください。")
    query = input("YOU: ")

    while len(query) > 0:
        buddy_res = practice_chain(query)
        conversation_str = get_conversation_string(practice_chain)
        critic_res = critic_chain.run({"conversation": conversation_str})

        print("FEEDBACK:\n" + critic_res)
        print("PRACTICE BUDDY: " + buddy_res["response"])
        query = input("YOU: ")


if __name__ == "__main__":
    main()