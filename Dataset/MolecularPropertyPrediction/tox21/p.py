import json

with open('/data1/bowei/agent/my_agent/Dataset/MolecularPropertyPrediction/bbbp/test.json','r',encoding='utf-8') as f:
    data = json.load(f)
n=0
for i in data:
    if i['gold_answer'] == 'Yes':
        n+=1

print(n)