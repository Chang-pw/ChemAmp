import requests
from rdkit import Chem
from ..Basemodel import ChatModel


def is_smiles(text):
    try:
        m = Chem.MolFromSmiles(text, sanitize=False)
        if m is None:
            return False
        return True
    except:
        return False

def largest_mol(smiles):
    ss = smiles.split(".")
    ss.sort(key=lambda a: len(a))
    while not is_smiles(ss[-1]):
        rm = ss[-1]
        ss.remove(rm)
    return ss[-1]

class SMILES2Description:
    name: str = "SMILES2Description"
    description: str = "Input only one molecule SMILES, returns Description. Note: the results returned by this tool may not necessarily be correct."
    def __init__(self, **tool_args):
        pass
    
    def _run(self, query: str,**tool_args) -> str:
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{}/{}"
        # Query the PubChem database
        r = requests.get(url.format(query, "Description/JSON"))
        # Convert the response to a JSON object
        data = r.json()
        try:
            smi = data['InformationList']['Information'][1]['Description']
        except:
            return "Could not find a molecule matching the text. One possible cause is that the input is incorrect, please modify your input."
    
        return str(smi)
    
    def __str__(self):
        return "SMILES2Description"

    def __repr__(self):
        return self.__str__()

    def wo_run(self,query,debug=False):
        model = ChatModel()
        prompt = "Please output only one molecule SMILES for use in generating Description based on the question:" + query
        response,all_tokens = model.chat(prompt=prompt,history=[])
        answer = self._run(response)
        if answer == "Could not find a molecule matching the text. One possible cause is that the input is incorrect, please modify your input.":
            return "",all_tokens
        return answer,all_tokens


