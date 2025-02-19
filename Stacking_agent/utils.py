import inspect
from datetime import datetime
import nltk.translate.bleu_score as bleu
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import DataStructs
from rouge_score import rouge_scorer
import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')
from nltk.translate.meteor_score import meteor_score
import random
from sklearn.metrics import confusion_matrix, accuracy_score

def function_to_json(func) -> dict:
    """
    Converts a Python function or method into a JSON-serializable dictionary
    that describes the function's signature, including its name, description, and parameters.
    
    We exclude **tool_args from the parameters.
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    # Check if it's a function
    if inspect.isfunction(func):
        signature = inspect.signature(func)
        tool_name = func.__name__
        tool_description = func.__doc__ or ""
    
    # Check if it's a method (bound to a class instance)
    elif isinstance(func, (staticmethod, classmethod)):
        signature = inspect.signature(func.__func__)
        tool_name = func.__func__.__name__
        tool_description = func.__doc__ or ""
    
    # Check if it's a method (instance method)
    elif inspect.ismethod(func):
        signature = inspect.signature(func)
        tool_name = func.__name__
        tool_description = func.__doc__ or ""
    
    # Check if it's a class (constructor method)
    elif inspect.isclass(func):
        # If it's a class, use the signature of the __init__ method
        init_method = func.__init__ if hasattr(func, '__init__') else None
        if init_method:
            signature = inspect.signature(init_method)
        else:
            signature = inspect.Signature()
        tool_name = func.__name__
        tool_description = func.__doc__ or ""
    
    else:
        raise TypeError(f"Provided input is neither a function, method, nor a class: {type(func)}")

    # Process the signature to extract parameters
    parameters = {}
    for param in signature.parameters.values():
        # Skip 'self' for instance methods and exclude **tool_args
        if param.name == 'self' or param.kind == inspect.Parameter.VAR_KEYWORD:
            continue

        param_type = type_map.get(param.annotation, "string")  # Default to "string"
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty and param.name != 'self' and param.kind != inspect.Parameter.VAR_KEYWORD
    ]

    # Return the function or method signature in a JSON serializable format
    return {
        "tool_name": tool_name,
        "tool_description": tool_description,
        "parameters": {
            "type": "object",
            "properties": parameters,
            "required": required,
        },
    }


def extract_instance_params(instance):
    """
    Extracts the parameters of an instance object by inspecting its __init__ method
    """
    init_signature = inspect.signature(instance.__init__)
    init_params = {}
    for param_name, param in init_signature.parameters.items():
        if param_name == 'self':
            continue
        param_value = getattr(instance, param_name, None)  
        if param_value not in [None, {}]:
            init_params[param_name] = param_value
            
    return init_params
    

def test(query:str):
    """
    This is a test function
    """
    return query

def calculate_exact(str1: str, str2: str, normalize: bool = True):
    if not str1 and not str2: 
        return 1.0
    if not str1 or not str2:  
        return 0.0

    if normalize:
        str1 = str1.lower().strip()
        str2 = str2.lower().strip()
    return 1.0 if str1 == str2 else 0.0

def calculate_dis(s1, s2):

    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(dp[i - 1][j] + 1,      # 删除
                           dp[i][j - 1] + 1,      # 插入
                           dp[i - 1][j - 1] + cost) # 替换
    return dp[m][n]

def calculate_Validity(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return int(mol is not None)  
    except:
        return 0
def calculate_FTS(smiles1,smiles2):
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if not mol1 or not mol2:  
            return 0, 0, 0
        maccs1 = AllChem.GetMACCSKeysFingerprint(mol1)
        maccs2 = AllChem.GetMACCSKeysFingerprint(mol2)
        tanimoto_maccs = DataStructs.FingerprintSimilarity(maccs1, maccs2)

        rdk_fp1 = Chem.RDKFingerprint(mol1)
        rdk_fp2 = Chem.RDKFingerprint(mol2)
        tanimoto_rdk = DataStructs.FingerprintSimilarity(rdk_fp1, rdk_fp2)

        morgan_fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
        morgan_fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
        tanimoto_morgan = DataStructs.FingerprintSimilarity(morgan_fp1, morgan_fp2)

        return tanimoto_maccs,tanimoto_rdk,tanimoto_morgan
    except:
        return 0,0,0

def calculate_avg(data,str):
    score = 0
    for i in data:
        score +=i[str]
    return score/len(data)

def calculate_BLEU(generated_summary, reference_summary, n):
    warnings.filterwarnings("ignore", category=UserWarning)

    # Tokenize the generated summary and reference summary
    generated_tokens = list(generated_summary)
    reference_tokens = list(reference_summary)

    # Calculate the BLEU score
    weights = [1.0 / n] * n  # Weights for n-gram precision calculation
    bleu_score = bleu.sentence_bleu([reference_tokens], generated_tokens, weights=weights)
    return bleu_score

def calculate_rouge(generated_summary, reference_summary):
    try:
        scorer = rouge_scorer.RougeScorer(['rouge2', 'rouge4', 'rougeL'], use_stemmer=True)
        scores = scorer.score(generated_summary, reference_summary)
        return scores['rouge2'][0], scores['rouge4'][0], scores['rougeL'][0]
    except:
        return 0,0,0
    
def calculate_meteor(generated_summary, reference_summary):
    try:
        meteor = meteor_score(reference_summary, generated_summary)
        return meteor
    except:
        return 0



def sorted_tools(list):
    return sorted(list, key=lambda x: x['score'], reverse=True)


def task2query(task:str):
    if task == "Molecule_Design":
        query="Question: Please try to infer the SMILES of this molecule:\nDescription:"
    elif task == "Molecule_captioning":
        query= "Given the molecular SMILES, your task is to provide the detailed description of the molecule. Please strictly follow the format (The molecule is ...), no other information can be provided.\nMolecule SMILES: "
    elif 'MolecularPropertyPrediction' in task:
        task_name = task.split('_')[1].strip()
        if task_name == 'bace':
            query="Given a molecular structure represented by a SMILES string, please analyze whether this compound can act as a BACE1 (Beta-site Amyloid Precursor Protein Cleaving Enzyme 1) inhibitor. Considering the molecule's structural features including molecular weight, atom composition, bond types, and functional groups to determine its potential BACE1 inhibitory activity. This prediction is crucial for identifying potential therapeutic agents for Alzheimer's disease treatment. Please provide your prediction as a simple Yes or No.\nSMILES:"
        elif task_name == 'bbbp':
            query="Given a molecular structure represented by a SMILES string, please analyze whether this compound can penetrate the blood-brain barrier (BBB). Considering the molecule's structural features including molecular weight, atom composition, bond types, functional groups, lipophilicity, and other physicochemical properties that influence BBB penetration. This prediction is crucial for developing drugs that need to reach the central nervous system. Please provide your prediction as a simple Yes  or No .\nSMILES:"
        elif task_name == 'clintox':
            query="Given a molecular structure represented by a SMILES string, please analyze whether this compound has received FDA approval or failed in clinical trials specifically due to toxicity issues. Considering the molecule's structural features including molecular weight, atom composition, bond types, functional groups, and potential toxicophores that might influence its safety profile. This prediction is crucial for early assessment of drug candidates' potential success in clinical trials. Please provide your prediction as a simple Yes or No .\nSMILES:"
        elif task_name == 'hiv':
            query="Given a molecular structure represented by a SMILES string, please analyze whether this compound has the ability to inhibit HIV replication. Considering the molecule's structural features including molecular weight, atom composition, bond types, functional groups, and other physicochemical properties that might influence its anti-HIV activity. This prediction is crucial for identifying potential therapeutic agents for HIV treatment. Please provide your prediction as a simple Yes or No.\nSMILES:"
        elif task_name == 'tox21':
            query="Given a molecular structure represented by a SMILES string, please analyze whether this compound acts as an agonist or antagonist of the nuclear receptor androgen receptor (NR-AR). Considering the molecule's structural features including molecular weight, atom composition, bond types, functional groups, and structural characteristics that determine its interaction with NR-AR. This prediction is crucial for understanding the compound's potential endocrine-related activities. Please provide your prediction as a simple Yes or No:\nSMILES:"
    elif task == "ReactionPrediction":
        query = "Given an incomplete chemical reaction equation in SMILES notation (format: reactants>>product, where multiple reactants are separated by dots '.'), predict and complete the missing products marked as '___'. The response should only contain the only one SMILES representation of the missing molecule, without any additional explanation (Note: Please only output only one final product).\nPlease Answer the quetion based on the following Chemical reaction equation:"

    description = "Input the original complete question ,returns the answer. Note:1.the results returned by this tool may not necessarily be correct. 2.The input question try to be the same with the original complete question."
    return query, description
