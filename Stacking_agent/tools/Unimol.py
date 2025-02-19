import lmdb
import numpy as np
import os
import pickle
import pandas as pd
import argparse

class UniMol:
    name: str = "UniMol"
    description: str = 'Input the question, returns answers. Note: Please input the SMILES representation in the form of "SMILES"'
    task: str = 'bace'

    def __init__(
        self,
        **tool_args
    ):  
        import json
        print("当前打开的数据集是"+UniMol.task)
        with open(f"./Result/Stacking/MolecularPropertyPrediction_{UniMol.task}/UniMolv2.json", 'r', encoding='utf-8') as f:
            data_test = json.load(f)
        with open(f"./Result/Stacking/MolecularPropertyPrediction_{UniMol.task}/UniMolv2_train.json", 'r', encoding='utf-8') as f:
            data_train = json.load(f)    
        data = data_test+data_train
        
        self.query_data = {i['SMILES']:"Yes" if i['predict_TARGET']==1 else "No" for i in data}
        
    def _run(self, query: str,**tool_args) -> str:
        smiles = query.split("SMILES:")[-1].strip()
        try:
            if smiles in self.query_data:
                return self.query_data[smiles],0
        except:
            print("error")

            return "Please input the 'SMILES'",0
        
    def __str__(self):
        return "UniMol"

    def __repr__(self):
        return self.__str__()
    
    def wo_run(self,query):
        smiles = query.split("SMILES:")[-1].strip()
        try:
            if smiles in self.query_data:
                return self.query_data[smiles],0
        except:
            print("error")
            return "Please input the 'SMILES' ",0
    @classmethod
    def set_task(cls, new_task: str):
        cls.task = new_task

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def lmdb2csv(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    txn = env.begin()
    keys = list(txn.cursor().iternext(values=False))
    result = []
    for idx in keys:
        datapoint_pickled = txn.get(idx)
        data = pickle.loads(datapoint_pickled)
        mol = {}
        mol["SMILES"] = data['smi']
        target = data['target'][0]
        if target == -1.0:
            pass
        else:
            mol["TARGET"] = target
            result.append(mol)
    
    df = pd.DataFrame(result)
    df.to_csv(lmdb_path.rsplit('.', 1)[0]+".csv", index=False)
    return lmdb_path.rsplit('.', 1)[0]+".csv"

def get_data(file_path):
    train_csv_path = lmdb2csv(os.path.join(file_path,"train.lmdb"))
    val_csv_path = lmdb2csv(os.path.join(file_path,"valid.lmdb"))
    test_csv_path = lmdb2csv(os.path.join(file_path,"test.lmdb"))
    df1 = pd.read_csv(train_csv_path)
    df2 = pd.read_csv(val_csv_path)
    df_merged = pd.concat([df1, df2], ignore_index=True)
    df_merged.to_csv(train_csv_path.rsplit('.', 1)[0]+".csv", index=False)
    return train_csv_path,test_csv_path



