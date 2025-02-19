import requests
import time

def get_ChemDFM(input_text): 
    data_to_send = {'input_text':input_text}
    attempt = 0
    max_retries = 10
    backoff_factor = 1
    while attempt < max_retries:
        response = requests.post("http://10.176.40.139:4040/generate", json=data_to_send)
        if response.status_code == 200:
            return response.json()[0]
        attempt += 1
        print(f"Attempt {attempt} failed with status code: {response.status_code}. Retrying...")
        time.sleep(backoff_factor * (2 ** (attempt - 1)))
    raise Exception(f"Request failed after {max_retries} attempts")


class ChemDFM():
    name: str = "ChemDFM"
    description: str = 'Input the question, returns answers. Note: Input the complete original question you have heard to the tool will result in better outcomes'
    def __init__(
        self,
        **tool_args
    ):
        super(ChemDFM, self).__init__()
    
    def _run(self, query: str,**tool_args) -> str:
        return get_ChemDFM(query)
    
    def __str__(self):
        return "ChemDFM"

    def __repr__(self):
        return self.__str__()
    
    def wo_run(self,query):
        return get_ChemDFM(query),0
    

