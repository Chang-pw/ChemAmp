from ..Basemodel import ChatModel


class FinalRefer_agent:
    def __init__(self):
        self.system_prompt = """You are a strategic planning and final integration agent. 
You will be given a graduate-level question and reasoning outputs from all other agents.
Your task is to integrate all the information into a single, cohesive answer with detailed reasoning and evidence.

Your final output should:
1. Summarize the contributions from all agents, highlighting key insights.
3. Provide the final answer with a clear and detailed explanation.
4. Conclude with the final answer on a new line with the format: "The final answer is 'SMILES'
"""
        self.model = ChatModel()

    def _run(self,prompt):
        response,all_tokens = self.model.chat(prompt=prompt,history=[],system_prompt=self.system_prompt)
        
        return response,all_tokens