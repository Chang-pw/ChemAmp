# Load model directly
import json
import re

def get_ChemT5_data():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    max_length = 512
    num_beams = 10

    tokenizer = AutoTokenizer.from_pretrained("./multitask-text-and-chemistry-t5-base-augm")
    model = AutoModelForSeq2SeqLM.from_pretrained("./my_agent/temp/multitask-text-and-chemistry-t5-base-augm")

    import json
    with open("./Dataset/ReactionPrediction/test.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    for i in data:
        instance = i['SMILES']
        input_text = f"Predict the product of the following reaction: {instance}"

        text = tokenizer(input_text, return_tensors="pt")
        output = model.generate(input_ids=text["input_ids"], max_length=max_length, num_beams=num_beams)
        output = tokenizer.decode(output[0].cpu())

        output = output.split(tokenizer.eos_token)[0]
        output = output.replace(tokenizer.pad_token,"")
        output = output.strip()

        i['answer'] = output

        with open("./Result/Stacking/ReactionPrediction/Text_ChemT5_test.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

# get_ChemT5_data()

def get_ChemT5(query):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    max_length = 512
    num_beams = 10

    tokenizer = AutoTokenizer.from_pretrained("./multitask-text-and-chemistry-t5-base-augm")
    model = AutoModelForSeq2SeqLM.from_pretrained("./multitask-text-and-chemistry-t5-base-augm")

    input_text = f"Caption the following molecule: {query}"

    text = tokenizer(input_text, return_tensors="pt")
    output = model.generate(input_ids=text["input_ids"], max_length=max_length, num_beams=num_beams)
    output = tokenizer.decode(output[0].cpu())

    output = output.split(tokenizer.eos_token)[0]
    output = output.replace(tokenizer.pad_token,"")
    output = output.strip()
    return output

# ['TextChemT5_0','SMILES2Description_2']
    
class TextChemT5:
    name: str = "TextChemT5"
    description: str = 'Input the question, returns answers. Note: Please input the SMILES representation in the form of "SMILES"'
    def __init__(
        self,
        **tool_args
    ):
        with open("./Result/Stacking/Molecule_captioning/Text_ChemT5.json", 'r', encoding='utf-8') as f:
            data_test = json.load(f)
        with open("./Result/Stacking/Molecule_captioning/Text_ChemT5_train.json", 'r', encoding='utf-8') as f:
            data_train = json.load(f)    
        data = data_test+data_train
        self.query_data = {i['SMILES']:i['answer'] for i in data}
        
    def _run(self, query: str,**tool_args) -> str:
        smiles = query.split("SMILES:")[-1].strip()
        try:
            if smiles in self.query_data:
                return self.query_data[smiles],0
        except:
            return get_ChemT5(query),0
        
    def __str__(self):
        return "TextChemT5"

    def __repr__(self):
        return self.__str__()
    
    def wo_run(self,query):
        smiles = query.split("SMILES:")[-1].strip()
        try:
            if smiles in self.query_data:
                return self.query_data[smiles],0
        except:
            return get_ChemT5(query),0