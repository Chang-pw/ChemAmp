import os
from openai import AzureOpenAI
from typing import List, Dict
import time
import random
import dashscope
from http import HTTPStatus
from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_ENDPOINT=os.getenv("AZURE_API_ENDPOINT")
AZURE_API_VERSION=os.getenv("AZURE_API_VERSION")

if not AZURE_API_KEY or AZURE_API_ENDPOINT or AZURE_API_VERSION:
    raise ValueError("Please set your API_KEY in file .env")


class ChatModel():
    def __init__(self, model="gpt-4o", temperature=0.7):
        self.model = model
        self.temperature = temperature
        os.environ["OPENAI_API_TYPE"] = "azure"
        self.api_keys = AZURE_API_KEY
        self.api_endpoints = AZURE_API_ENDPOINT
    def chat(self, prompt: str, history: List[Dict[str, str]], system_prompt: str = 'You are a helpful assistant',stop_word:str='') -> str:
        """
        Get response with the prompt,history and system prompt.

        Args:
            prompt (str)
            history (List[Dict[str, str]])
            system_prompt (str)

        """
        total_tokens =0 
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for entry in history:
            messages.append(entry)
        messages.append({"role": "user", "content": prompt})

        chosen_index = random.randint(0, len(self.api_keys) - 1)
        chosen_key = self.api_keys[chosen_index]
        chosen_endpoint = self.api_endpoints[chosen_index]
        
        client = AzureOpenAI(
            api_key=chosen_key,
            api_version=AZURE_API_VERSION,
            azure_endpoint=chosen_endpoint
        )
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stop = stop_word
            )
        except Exception as e:
            print(e)
            return "Run Again",history
        total_tokens =0
        total_tokens = response.usage.total_tokens
        response = response.choices[0].message.content
            
        history.append({"role": "assistant", "content": response})
        try:
            response = response.replace("Thought:","\033[92mThought:\033[0m")
            response = response.replace("Action:","\033[93mAction:\033[0m")
            response = response.replace("Action Input:","\033[94mAction Input:\033[0m")
        except:
            pass
        try:
            response = response.replace("Final Answer:","\033[91mFinal Answer:\033[0m")
        except:
            pass
        
        return response,total_tokens
    
    

