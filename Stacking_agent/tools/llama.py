from http import HTTPStatus
import os
import dashscope
import requests
import time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def get_Llama_local(query,tokenizer,llm):

    messages = [
        {"role": "system", "content": "You are a chemistry assistant. Note: Please only output the final answer, no other information and explaination."},
        {"role": "user", "content": query}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    sampling_params = SamplingParams(temperature=0, top_p=0.8, repetition_penalty=1.05, max_tokens=1000)

    try:
        outputs = llm.generate([text],sampling_params=sampling_params)
    except Exception as e:
        print(f"Error occurred: {e}")    
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text

    return generated_text

def get_Llama_api(query):
    messages = [{'role': 'system', 'content': 'You are a chemistry assistant. Note: Please only output the final answer, no other information and explaination.'},
                {'role': 'user', 'content': query}]
    response = dashscope.Generation.call(
        api_key='Your ali-api',
        model='llama3-8b-instruct',
        messages=messages,
        result_format='message',  # set the result to be "message" format.
    )
    time.sleep(10)
    if response.status_code == HTTPStatus.OK:
        return response['output']['choices'][0]['message']['content']
    else:
        return "Error"
    
# print(get_Llama_api('你好'))

class Llama():
    name: str = "Llama"
    
    def __init__(
        self,
        **tool_args
    ):
        # self.t = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
        # self.l = LLM(model=model_path,trust_remote_code=True)

        super(Llama, self).__init__()

    # def _run(self, query: str,**tool_args) -> str:
    #     return get_Llama(query)
    
    def __str__(self):
        return "Llama"

    def __repr__(self):
        return self.__str__()
    
    def wo_run(self,query):
        return get_Llama_api(query),0
    