from ..Basemodel import ChatModel

Molecule_Design_prompt="""You are a Chemistry specialist.
You will be given a question related to chemistry.
Your task is to carefully analyze the question, extract key concepts, and provide a detailed explanation based on your expertise in chemistry. 
Focus only on principles, theories, and processes directly relevant to chemistry, unless the question explicitly mentions interdisciplinary aspects.
If the question involves calculations, data interpretation, or experimental information, ensure your reasoning is precise and well-structured. 
If you are unsure about an answer, state your reasoning clearly and indicate any knowledge gaps.

Your output should include:
1. A step-by-step explanation of your reasoning.
2. The final answer on a new line with the format: "The answer is 'SMILES'".
"""

Examples = """Question: Given the following molecule description, answer the molecule SMILES:
Description:The molecule is a piperidinemonocarboxylic acid in which the carboxy group is located at position C-2. It is a conjugate acid of a pipecolate.

This SMILES describes a piperidine ring with a carboxylic acid (-COOH) group attached to the second carbon in the ring.
The molecule described is 2-piperidinecarboxylic acid, commonly known as pipecolic acid. Its SMILES (Simplified Molecular Input Line Entry System) representation is:OC(=O)C1CCNCC1

The answer is 'OC(=O)C1CCNCC1'
"""

No_tool_prompt = Molecule_Design_prompt + "\nHere is the example:" + Examples 


class Chat_agent:
    def __init__(self):
        self.system_prompt = No_tool_prompt
        self.model = ChatModel()

    def _run(self,prompt):
        response,all_tokens = self.model.chat(prompt=prompt,history=[],system_prompt=self.system_prompt)
        
        return response,all_tokens