from http import HTTPStatus
import dashscope
import requests
import time
from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)
ALI_API_KEY = os.getenv("ALI_API_KEY")

if not ALI_API_KEY:
    raise ValueError("Please set your API_KEY in file .env")


def get_Llama_api(query):
    messages = [{'role': 'system', 'content': 'You are a chemistry assistant. Note: Please only output the final answer, no other information and explaination.'},
                {'role': 'user', 'content': query}]
    response = dashscope.Generation.call(
        api_key=ALI_API_KEY,
        model='llama3-8b-instruct',
        messages=messages,
        result_format='message',  # set the result to be "message" format.
    )
    time.sleep(10)
    if response.status_code == HTTPStatus.OK:
        return response['output']['choices'][0]['message']['content']
    else:
        return "Error"
    

class Llama():
    name: str = "Llama"
    
    def __init__(
        self,
        **tool_args
    ):
        super(Llama, self).__init__()

    # def _run(self, query: str,**tool_args) -> str:
    #     return get_Llama(query)
    
    def __str__(self):
        return "Llama"

    def __repr__(self):
        return self.__str__()
    
    def wo_run(self,query):
        return get_Llama_api(query),0
    