import requests
from rdkit import Chem


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

def extract_property(data):
    compound = data['PC_Compounds'][0]
    properties = {}
    for prop in compound['props']:
        label = prop['urn']['label']
        name = prop['urn'].get('name', '')
        if label in ['SMILES','IUPAC Name','Fingerprint','InChI','InChIKey']:
            continue
        # 获取值
        value = next((v for k, v in prop['value'].items()), None)

        # 创建属性键名
        key = f"{label}_{name}" if name else label
        properties[key] = value
    return properties


class SMILES2Property:
    name: str = "SMILES2Property"
    description: str = "Input one SMILES, returns Property. "
    def __init__(self, **tool_args):
        pass
    
    def _run(self, query: str,**tool_args) -> str:
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{}/{}"
        # Query the PubChem database
        r = requests.get(url.format(query, "/JSON"))
        # Convert the response to a JSON object
        data = r.json()
        try:
            smi = data
        except:
            return "Could not find a molecule matching the text. One possible cause is that the input is incorrect, please modify your input."
        try:
            smi = extract_property(smi)
        except:
            pass
        return str(smi)
    
    def __str__(self):
        return "SMILES2Property"

    def __repr__(self):
        return self.__str__()

    def wo_run(self,query,debug=False):
        
        return "",0
    
if __name__ == "__main__":
    tool = SMILES2Property()
    print(tool._run("CC(=O)Nc1ccc(cc1)N2CCN(CC2)C3=CC=CC=C3"))