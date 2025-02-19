from concurrent.futures import ProcessPoolExecutor
import argparse
import json
from Stacking_agent.Stacking import *
from Stacking_agent.tools import *
from Stacking_agent.utils import *
from Stacking_agent.generator import *
from datetime import datetime
import time

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_stacking(tdn, file_path, tools, topN, tool_number, task, task_query):
    train_data = load_data(file_path)
    result, _ = Stacking(tools=tools, top_n=topN, tool_number=tool_number, train_data=train_data,
                         train_data_number=tdn, task=task, query=task_query)._run()
    return {'train_data_number': tdn, 'result': str(result)}

def ablation(task, topN, tool_number, tools, task_query):
    all_result = []
    print(task)
    if any(keyword in task for keyword in ["MolecularPropertyPrediction"]):
        task_name = task.split('_')[-1]
        task_ = task.split('_')[0]
        file_path = f'./Dataset/{task_}/{task_name}/train.json'
    else:
        file_path = f'./Dataset/{task}/train.json'

    train_data_number = [5, 10, 20, 30]

    # 使用ProcessPoolExecutor进行并行处理
    with ProcessPoolExecutor() as executor:
        all_result = list(executor.map(
            run_stacking,
            train_data_number,
            [file_path] * len(train_data_number),
            [tools] * len(train_data_number),
            [topN] * len(train_data_number),
            [tool_number] * len(train_data_number),
            [task] * len(train_data_number),
            [task_query] * len(train_data_number)
        ))

    return all_result

def main():
    parser = argparse.ArgumentParser(description="The ablation experiment")
    parser.add_argument('--Task', type=str, help="The chemical task for Agent", required=True)
    parser.add_argument('--tools', type=str, help="The tools for Agent", required=True)
    parser.add_argument('--topN', type=int, help="The top N of the tools for stacking", required=True)
    parser.add_argument('--tool_number', type=int, help="The number of tools for one Agent", required=True)

    args = parser.parse_args()

    task = args.Task
    tools = args.tools
    topN = args.topN
    tool_number = args.tool_number

    tools = eval(tools)

    task_query, task_description = task2query(task)
    Agent_tool.set_description(task_description)
    Agent_tool.set_task_name(task)

    result = ablation(task, topN, tool_number, tools, task_query)
    now = datetime.now()

    with open(f'./Result/ablation/{task}_{topN}_{tool_number}_{now}.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

if __name__ == '__main__':
    main()
