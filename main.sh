## Task: 'Molecule Generation'='Query2SMILES';'Molecule Caption'='SMILES2Query'
## Tools: 'Name2SMILES','ChemDFM',
## TopN: Selection of the best performing N tools for stacking
## Tool Number: Number of tools each agent can call
## Train data number: Number of training data
topN=5
tool_number=2
train_data_number=20

tools='''
[
SMILES2Property(),
Chemformer(),
]
'''
# Task='ReactionPrediction'
# python main.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"