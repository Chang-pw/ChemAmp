from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)
ALI_API_KEY = os.getenv("ALI_API_KEY")
if not ALI_API_KEY:
    raise ValueError("Please set your API_KEY in file .env")




def get_deepseek(query):
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=ALI_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="deepseek-r1", # 此处以 deepseek-r1 为例，可按需更换模型名称。
        messages=[
            {"role": "system", "content": "You are a chemistry assistant. Note: Please only output the final answer, no other information and explaination."},
            {"role": "user", "content": query +"Note: Please only output the final answer, no other information and explaination."}
            ],
        stream=True
        )

    reasoning_content = ""
    # 定义完整回复
    answer_content = ""
    for chunk in completion:
        # 获取思考过程
        reasoning_chunk = chunk.choices[0].delta.reasoning_content
        answer_chunk = chunk.choices[0].delta.content
        if reasoning_chunk != "":
            # print(reasoning_chunk,end="")
            reasoning_content += reasoning_chunk
        elif answer_chunk != "":
            # print(answer_chunk,end="")
            answer_content += answer_chunk
    return answer_content

# print(get_deepseek(''))
class Deepseek():
    name: str = "Deepseek"
    description: str = ""
    def __init__(
        self,
        **tool_args
    ):
        super(Deepseek, self).__init__()

    # def _run(self, query: str,**tool_args) -> str:
    #     return get_Llama(query)
    
    def __str__(self):
        return "Deepseek"

    def __repr__(self):
        return self.__str__()
    
    def wo_run(self,query):
        try:
            return get_deepseek(query),0
        except:
            return "Error",0
