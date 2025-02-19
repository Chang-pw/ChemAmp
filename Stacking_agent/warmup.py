from .agent import Agent
from .utils import *
from .tools import *
import random
import json
import time
from tqdm import tqdm
import concurrent.futures
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc


class Warmup:
    def __init__(self,tools=[],tool_number=2,data=[],train_data_number=10,task="",query=""):
        self.seed = 2025
        self.tool = tools
        self.tool_number = tool_number
        self.data = data
        self.train_data_number = train_data_number
        self.task = task
        self.task_query = query
        self.sample_data = self.sample()
        self.debug = False
    def sample(self):
        # random.seed(self.seed)
        sample_data = random.sample(self.data,self.train_data_number)
        if 'MolecularPropertyPrediction' in self.task:
            if len([i for i in sample_data if i['gold_answer']=='Yes']) >= 0:
                p_data = [i for i in self.data if i ['gold_answer']=='Yes']
                add_data = random.sample(p_data,3)
                sample_data.append(add_data[0])

        return sample_data

    def test(self,tool=[],wo_agent=False):
        agent = Agent(tool)
        sample_data = self.sample_data
        score = 0
        if self.task in ['Molecule_Design', 'Molecule_captioning']:
            for index,i in enumerate(sample_data):
                smiles = i["SMILES"]
                description = i["description"]
                
                if self.task =='Molecule_Design':
                    query = self.task_query + description
                    reference = smiles
                else:
                    query = self.task_query + smiles
                    reference = description

                if wo_agent:
                    final_answer,all_tokens = tool[0].wo_run(query)
                    agent = tool[0]
                else:
                    final_answer, response, history,all_tokens = agent._run(query,[],debug=self.debug,index=index)
                i["answer"] = final_answer
                i['all_tokens'] = all_tokens

                i["blue2"] = calculate_BLEU(final_answer,reference,2)
                score += i["blue2"]
                time.sleep(5)
            score = score/len(sample_data)

        elif 'MolecularPropertyPrediction' in self.task:
            for index,i in enumerate(sample_data):
                smiles = i["SMILES"]
                gold_answer = i['gold_answer']
                query = self.task_query + smiles
                if wo_agent:
                    final_answer,all_tokens = tool[0].wo_run(query)
                    agent = tool[0]
                else:
                    final_answer, response, history,all_tokens = agent._run(query,[],debug=self.debug,index=index)
                if "Yes" in final_answer:
                    final_answer = "Yes"
                else:
                    final_answer = "No"
                i["answer"] = final_answer
                i['all_tokens'] = all_tokens

                time.sleep(5)
            y_true = [1 if i['gold_answer']=='Yes' else 0 for i in sample_data]
            y_pred = [1 if i['answer']=='Yes' else 0 for i in sample_data]
            score = accuracy_score(y_true, y_pred,zero_division=1.0)

    

        elif self.task == 'ReactionPrediction':
            for index,i in enumerate(sample_data):
                smiles = i["SMILES"]
                reaction = i["reaction"]
                query = self.task_query + reaction
                if wo_agent:
                    final_answer,all_tokens = tool[0].wo_run(query)
                    agent = tool[0]
                else:
                    final_answer, response, history,all_tokens = agent._run(query,[],debug=self.debug,index=index)
                i["answer"] = final_answer
                i["blue2"] = calculate_BLEU(final_answer,smiles,2)
                score += i["blue2"]
                i['all_tokens'] = all_tokens

                time.sleep(5)
            score = score/len(sample_data)

        return agent,score,sample_data

    def one_tool_stacking(self,tool:dict):
        name = str(tool[0])
        layer = 0
        score = -1
        Tool_agent= []
        tool_list = tool
        Score_list = []
        while True:
            if layer == 0:
                wo_agent = True
                test_agent,blue2,sample_data = self.test(tool_list,wo_agent=wo_agent)
            elif layer == 1:
                wo_agent = False
                test_agent,blue2,sample_data = self.test(tool_list)
            else:
                Tool_agent_list = Tool_agent[1:][-(self.tool_number-1):]
                test_agent,blue2,sample_data = self.test(tool_list+Tool_agent_list)
            print(f"The score of the {layer} layer of the {name} stack is {blue2}")
            if blue2 > score:
                with open(f"./Result/Stacking/{self.task}/warmup_{name}_{layer}.json","w",encoding="utf-8") as f:
                    json.dump(sample_data,f,indent=4)
                score = blue2
                if wo_agent:
                    Agent_t = tool[0]
                else:
                    Agent_t = Agent_tool(test_agent,data=sample_data)
                Tool_agent.append(Agent_t)
                Score_list.append(blue2)
                layer +=1
                if score == 1:
                    break
            else:
                break
        return Tool_agent,Score_list

    def _run(self):
        tool_list = self.tool
        result_list = []
        print("\033[31m ----WarmUp Start---- \033[0m\n")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.one_tool_stacking, [tool]) for tool in tool_list]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    tool_agent, score_list = future.result()
                    
                    tool_name = tool_list[futures.index(future)]
                    print(f"{tool_name} Stacking success")
                    print(f"{tool_name} 's best performance is {score_list[-1]}")
                    
                    layer = 0
                    for agent, score in zip(tool_agent, score_list):
                        result_list.append({"agent_tool": agent, "score": score, "tool": f"{tool_name}_{layer}"})
                        layer += 1

                except Exception as e:
                    print(f"Erro with tool call: {e}")


        print("\n\033[31m ----WarmUp End---- \033[0m")
        print("\n\033[34mAvilable Tools：\033[0m")
        for index,i in enumerate(result_list):
            print(f"{index+1}:{i}")
        return result_list
    