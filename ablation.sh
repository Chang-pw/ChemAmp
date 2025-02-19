Task='Molecule_Design'
topN=5
tool_number=2
tools='''
[
Name2SMILES(),
ChemDFM()
]
'''
python ablation.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number"
# python ablation.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number"&
# python ablation.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number"&

# tool_number=3
# python ablation.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number"
# tool_number=4
# python ablation.py  --Task "$Task" --tools "$tools"  --topN "$topN" --tool_number "$tool_number"
