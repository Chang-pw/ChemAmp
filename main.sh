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
tools='''
[
SMILES2Property(),
UniMol(),
]
'''
# Task='MolecularPropertyPrediction_bace'
# python main.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"
Task='MolecularPropertyPrediction_bace'
python main.py  --no_train --Stacking "['UniMol_0','SMILES2Property_0']" --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"
python main.py  --no_train --Stacking "['UniMol_1']" --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"
python main.py  --no_train --Stacking "[['UniMol_1', 'SMILES2Property_0'], 'SMILES2Property_0']" --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"
python main.py  --no_train --Stacking "['UniMol_0','SMILES2Property_1']" --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"


# Task='MolecularPropertyPrediction_hiv'
# python main.py   --no_train --Stacking "['UniMol_0', 'SMILES2Property_1']" --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"

# python main.py --no_train --Stacking "['UniMol_0','SMILES2Property_0']" --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"&
# Task='MolecularPropertyPrediction_clintox'
# python main.py --no_train --Stacking "[['UniMol_1', 'SMILES2Property_0'], ['UniMol_1', 'SMILES2Property_1']]" --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"
# Task='Molecule_Design'
# python main.py --no_train --Stacking "['Name2SMILES_0','ChemDFM_0','SMILES2Description_0','TextChemT5_0','UniMol_0','SMILES2Property_0']" --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"
# Task='Molecule_captioning'
# python main.py --no_train --Stacking "Llama3_0" --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"
# Task='ReactionPrediction'
# python main.py --no_train --Stacking "['Chemformer_2','SMILES2Property_0']" --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"


# python main.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"

# python main.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"
# tools='''
# [
# SMILES2Property(),
# ChemDFM()
# ]
# '''
# Task='ReactionPrediction'
# python main.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"&
# python main.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"

# Task='Retrosynthesis'
# python main.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"&
# python main.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"

# Task='ReactionPrediction'
# python main.py   --no_train --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"
# python main.py  --no_train --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"
# python main.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"

# Task='ReagentSelection_solvent'
# python main.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"&
# python main.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"

# Task='ReagentSelection_reactant'
# python main.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"&
# python main.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"





# Task='ReagentSelection_solvent'
# python main.py --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"

# Task='ReagentSelection_ligand'
# python main.py --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"

# Task='ReagentSelection_reactant'
# python main.py --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"

# Task='ReagentSelection_solvent'
# python main.py --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"

# Task='ReagentSelection_ligand'
# python main.py --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number" --train_data_number "$train_data_number"