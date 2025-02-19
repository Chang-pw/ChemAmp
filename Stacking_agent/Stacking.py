from .warmup import *
from .agent import Agent
from .utils import calculate_BLEU,sorted_tools
from .tools import *
import random
import json
import time
from tqdm import tqdm
import itertools
import dill
import math
import re
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc

class Stacking:
    def __init__(self,tools:list,top_n:int,tool_number=2,train_data=[],train_data_number=10,task="",query=""):
        self.all_tools = tools
        self.tool_number = tool_number
        self.task = task
        self.query = query
        self.Warmup = Warmup(self.all_tools,tool_number=self.tool_number,data=train_data,train_data_number=train_data_number,task=self.task,query=self.query)
        self.data = self.Warmup.sample_data
        self.warmup = sorted_tools(self.Warmup._run())
        self.top_n = top_n
        self.debug = False


    def test(self,tool_list,name1,name2):
        test_agent = Agent(tool_list)
        test_data = self.data
        score =0 
        if self.task in ['Molecule_Design', 'Molecule_captioning']:
            for i in tqdm(test_data):
                smiles = i['SMILES']
                description = i['description']
                if self.task =='Molecule_Design':
                    query = self.query + description
                    reference = smiles
                else:
                    query = self.query + smiles
                    reference = description
                final_answer, response, history,all_tokens = test_agent._run(query,[],debug=self.debug)
                i['answer'] = final_answer
                i['all_tokens'] = all_tokens
                i['blue2'] = calculate_BLEU(final_answer,reference,2)
                time.sleep(5)
                score += i['blue2']
            
            score = score/len(test_data)
            
        elif 'MolecularPropertyPrediction' in self.task:
            for i in tqdm(test_data):
                smiles = i['SMILES']
                gold_answer = i['gold_answer']
                query = self.query + smiles
                final_answer, response, history,all_tokens = test_agent._run(query,[],debug=self.debug)
                if "Yes" in final_answer:
                    final_answer = "Yes"
                else:
                    final_answer = "No"
                i["answer"] = final_answer
                i['all_tokens'] = all_tokens


            y_true = [1 if i['gold_answer']=='Yes' else 0 for i in test_data]
            y_pred = [1 if i['answer']=='Yes' else 0 for i in test_data]
            score = accuracy_score(y_true, y_pred,zero_division=1.0)

        elif self.task=='ReactionPrediction':
            for i in tqdm(test_data):
                smiles = i['SMILES']
                reaction = i['reaction']
                query = self.query + reaction
                final_answer, response, history,all_tokens = test_agent._run(query,[],debug=self.debug)
                i['answer'] = final_answer
                i['blue2'] = calculate_BLEU(final_answer,smiles,2)
                i['all_tokens'] = all_tokens

                time.sleep(5)
                score += i['blue2']

            score = score/len(test_data)

        return test_agent,score,test_data


    def one_Stacking(self,tool_list):
        """ Note that Input the Sorted tool list """
        # Top N tools: tools
        tools = tool_list[:self.top_n]
        # Best performance tool: tool_1
        tool_1 = tools[0]
        # Other tools for stacking : tools 
        remaining_tools = tools[1:]
        # Result list
        result_list = []

        tool_number = self.tool_number
        if tool_number > len(tools):
            print(f"Since tool_number is greater than the topN tool list, set tool_number to the maximum value {len(tools)}.\n")
            tool_number = len(tools)
        try:
            for i in remaining_tools:
                match = re.match(r'^(.*)_\d+$', tool_1['tool'])
                if i['tool'] == f'{match.group(1)}_0':
                    remaining_tools.remove(i)
                    print(f"Since the required tool {tool_1['tool']} has been stacking with the tool {match.group(1)}_0 during the warm-up, the base tool will be excluded\n")
        except:
            pass
        
        tool_combinations = itertools.combinations(remaining_tools, tool_number - 1)
        combination_number = math.comb(len(remaining_tools), tool_number - 1)

        print(f"The required tool is {tool_1['tool']}, and the remaining top{self.top_n} tools are {[i['tool'] for i in remaining_tools]}\n")

        print(f"{tool_number - 1} tools will be selected from the remaining tools and combined with {tool_1['tool']}, resulting in a total of {combination_number} combinations.\n")

        for combination in tool_combinations:
            # Combine tool_1 with the remaininged cosmbination of tools
            tool_combination = [tool_1['agent_tool']] + [tool['agent_tool'] for tool in combination]
            
            # Get the names of the tools in the current combination
            tool_names = [tool_1['tool']] + [tool['tool'] for tool in combination]

            print(f"The current stacking tool combination is: {tool_names}")

            # Call the test function with the current combination of tools
            test_agent, blue2 ,sample_data= self.test(tool_combination, tool_1['tool'], tool_names[-1])
            try:
                with open(f'./Result/Stacking/{self.task}/Stacking_{tool_names}.json','w',encoding='utf-8') as f:
                    json.dump(sample_data,f,indent=4)
            except:
                with open(f'./Result/Stacking/{self.task}/Stacking_error.json','w',encoding='utf-8') as f:
                    json.dump(sample_data,f,indent=4)

            # Append the result to the result list
            result_list.append({'agent_tool': test_agent, 'score': blue2, 'tool':tool_names})
            print(f"The score for the current stacking tool combination is: {blue2}")
        result_list = sorted_tools(result_list)
        return result_list
    def _run(self):
        tool_list = self.all_tools
        warmup_result_list = self.warmup
        top_score = warmup_result_list[0]['score']
        layer = 1
        only_one = False
        result_list = []
        print('\n\033[31m ----Stacking start---- \033[0m\n')
        while True:
            print(f'\033[34m --The Current Stacking Layer is {layer}-- \033[0m')
            if layer == 1:
                last_result_list = warmup_result_list
                if len(last_result_list) == 1:
                    print('Since there is only one tool in the warm-up, end the stacking')
                    only_one = True
                    break
                if len(last_result_list) == 2 and len(tool_list) == 1:
                    print('Since there is only one tool stack in the warmup and only one tool is stacking, the tool stacking and the second layer of the warmup will be repeated, end the stacking')
                    only_one = True
                    break
                result_list = self.one_Stacking(last_result_list)
            else:
                last_result_list = result_list
                result_list = self.one_Stacking(last_result_list)
                
            layer +=1

            if result_list[0]['score'] <= top_score:
                print(f'The stacking score of the {layer} is lower than the highest score of the previous layer, and the stacking ends.')
                result_list = sorted_tools(last_result_list)
                break
            top_score = result_list[0]['score']
            print(f'The highest score of the {layer} layer is {top_score}, end the stacking and go to the next layer')

            for i in result_list:
                with open(f'./Result/Stacking/{self.task}/Stacking_{i["tool"]}.json','r',encoding='utf-8') as f:
                    sample_data = json.load(f)
                i['agent_tool'] = Agent_tool([i['agent_tool']],data=sample_data)

            result_list = last_result_list + result_list
            result_list = sorted_tools(result_list)

        print('\n\033[31m ----Stacking End---- \033[0m\n')
        if only_one:
            result_list = sorted_tools(warmup_result_list)
        print("\n\033[34mFinal Stacking result is：\033[0m")
        for index,i in enumerate(result_list):
            print(f"{index+1}:{i}")

        return result_list,top_score
    