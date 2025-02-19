import json
import pandas as pd
import csv

# df = pd.read_csv('/data1/bowei/agent/my_agent/Result/Stacking/MolecularPropertyPrediction_clintox/clintox_test.predict.0.csv')

# df.to_json('/data1/bowei/agent/my_agent/Result/Stacking/MolecularPropertyPrediction_clintox/UniMolv2.json', orient='records', lines=False)

# with open('/data1/bowei/agent/my_agent/Result/Stacking/MolecularPropertyPrediction_bbbp/UniMolv2.json', 'r',encoding='utf-8') as file:
#     df = json.load(file)

# new= []
# for i in df:
#     new.append({'SMILES':i['SMILES'],'gold_answer':"Yes" if i['TARGET']==1 else "No",'answer':"Yes" if i['predict_TARGET']==1 else "No"})
# with open('/data1/bowei/agent/my_agent/Result/Stacking/MolecularPropertyPrediction_bbbp/UniMolv2.json', 'w',encoding='utf-8') as file:
#     json.dump(new,file)

# 读取 JSON 文件
task = "tox21"
ls = 'all'
json_file = f'Dataset/MolecularPropertyPrediction/{task}/{ls}.json'  # 替换为你的 JSON 文件路径
with open(json_file, 'r',encoding='utf-8') as file:
    df = json.load(file)
# print(df)
data = []
for i in df:
    data.append({'SMILES':i['SMILES'],'TARGET':1 if i['gold_answer']=='Yes' else 0})

fieldnames = data[0].keys()

# 打开文件并使用 DictWriter 写入数据
with open(f'Dataset/MolecularPropertyPrediction/{task}/{task}_{ls}.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    
    # 写入列名（表头）
    writer.writeheader()
    
    # 写入字典中的每一行
    writer.writerows(data)