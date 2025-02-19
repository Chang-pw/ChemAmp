import asyncio
import random
from typing import List, Dict, Tuple
from Stacking_agent.Stacking import *
from tqdm import tqdm
from Stacking_agent.prompt import *
from Stacking_agent.tools import *
from Stacking_agent.utils import *
from Stacking_agent.generator import *
import json
import os


# --------------------- Agent Setting ---------------------
task_query,task_description = task2query("Molecule_Design")
Agent_tool.set_description(task_description)
Agent_tool.set_task_name("Molecule_Design")
generator = ToolGenerator()
Final_agent,wo = generator.generate("['ChemDFM_0','Name2SMILES_0']")
FinalRefer_node = FinalRefer_agent()
FinalRefer_prompt ="""
Here is the question:{question}. At the same time, the output of other agents is as follows:
{answers}
"""

# --------------------- topology ---------------------
def generate_layered_graph(N: int, layer_num: int = 2) -> List[List[int]]:
    """generate topology with layered structure"""
    adj_matrix = [[0]*N for _ in range(N)]
    base_size = N // layer_num
    remainder = N % layer_num
    
    # Layer
    layers = []
    for i in range(layer_num):
        size = base_size + (1 if i < remainder else 0)
        layers.extend([i]*size)
    random.shuffle(layers)
    
    # Connect
    for i in range(N):
        current_layer = layers[i]
        for j in range(N):
            if layers[j] == current_layer + 1:
                adj_matrix[i][j] = 1
    return adj_matrix

def generate_star_graph(N: int) -> List[List[int]]:
    """Star"""
    matrix = [[0]*N for _ in range(N)]
    for i in range(1, N):
        matrix[0][i] = 1  
        matrix[i][0] = 1  
    return matrix

# --------------------- Mask ---------------------
def generate_masks(mode: str, N: int) -> Tuple[List[List[int]], List[List[int]]]:
    if mode == 'FullConnected':
        spatial = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        temporal = [[1]*N for _ in range(N)]  
    
    elif mode == 'Random':
        spatial = [[random.randint(0, 1) if i != j else 0 for i in range(N)] for j in range(N)]
        temporal = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    
    elif mode == 'Chain':
        spatial = [[1 if i == j+1 else 0 for i in range(N)] for j in range(N)]
        temporal = [[1 if step == j else 0 for j in range(N)] for step in range(N)] 
    elif mode == 'Debate':
        spatial = [[0]*N for _ in range(N)]  
        temporal = [[1]*N for _ in range(N)]  
    
    elif mode == 'Layered':
        spatial = generate_layered_graph(N)
        temporal = [[1]*N for _ in range(N)]  
        print(spatial, temporal)

    elif mode == 'Star':
        spatial = generate_star_graph(N)
        temporal = [[1]*N]*3  
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return spatial, temporal

# --------------------- Main ---------------------
async def run_agent(agent: Agent, prompt: str,no_tool=False) -> str:
    print(prompt)
    if not no_tool:
        return await asyncio.get_event_loop().run_in_executor(None, agent.test_run, prompt)
    else:
        chat = Chat_agent()
        return await asyncio.get_event_loop().run_in_executor(None, chat._run, prompt)

async def simulate(
    agents: List[Agent],
    initial_prompt: str,
    mode: str,
    max_steps: int = None,
    no_tool: bool = False
) -> Dict[int, List[str]]:
    N = len(agents)
    spatial_masks, temporal_masks = generate_masks(mode, N)
    message_queues = {i: [] for i in range(N)}
    history = {i: [] for i in range(N)}
    sum : int = 0 
    
    # Max steps
    if max_steps is None:
        max_steps = len(temporal_masks) if mode in ['Star'] else N*2
    
    for step in range(max_steps):
        # Dynamic temporal mask
        if step >= len(temporal_masks):
            if mode == 'Chain':
                temporal_mask = [0]*N
            else:
                temporal_mask = [1]*N 
        else:
            temporal_mask = temporal_masks[step]
        
        # Agent
        active_agents = [j for j in range(N) if temporal_mask[j]]

        # Prompt
        current_prompts = []
        for i in range(N):
            prompt = initial_prompt
            # Add message
            if message_queues[i]:
                prompt += "\nYou will be given a graduate-level question and the reasoning outputs from other agents. Note that the references answers may not be correct, please have a critique of the reasoning process for each agent’s output. These are previous responses for reference:" + "\n".join(message_queues[i])
                message_queues[i] = []
            # Debate mode
            if mode == 'Debate' and step > 0:
                references = "\nPrevious Responses:\n" + "\n".join(
                    [f"Agent {j}: {history[j][-1]}" for j in range(N)]
                )
                prompt += references + "\nCritique each analysis above."
            current_prompts.append(prompt)
        
        # async run
        tasks = [run_agent(agents[j], current_prompts[j],no_tool=no_tool) for j in active_agents]
        answers = await asyncio.gather(*tasks)

        for idx, j in enumerate(active_agents):
            answer,all_tokens = answers[idx]
            sum+=all_tokens
            history[j].append(answer)
            # Spatial mask
            for i in range(N):
                if spatial_masks[j][i]:
                    message_queues[i].append(f"Agent {j}: {answer}")

    return history,sum

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-agent topology experiment")
    parser.add_argument("--mode", type=str, required=True, 
                        choices=['FullConnected', 'Random', 'Chain', 'Debate', 'Layered', 'Star'],
                        help="mode")
    parser.add_argument("--agents", type=int, default=2, help="Agent number")
    parser.add_argument("--steps", type=int, default=None, help="max steps")
    parser.add_argument("--no_tool", type=bool, default=False, help="tool or not")
    args = parser.parse_args()  

    agent_numbers = args.agents

    if not args.no_tool:
        filepath = f'./Result/Multiagent/Tool/{args.mode}_{agent_numbers}_{args.steps}.json'
    else:
        filepath = f'./Result/Multiagent/NoTool/{args.mode}_{agent_numbers}_{args.steps}.json'
    if agent_numbers==0:
        agent_numbers = 1

    agents = [Final_agent for _ in range(agent_numbers)]
    if os.path.exists(filepath):
        with open(filepath,'r',encoding='utf-8') as f:
            new_data = json.load(f)
        len_ = len(new_data)
    else:
        new_data=[]
        len_=0
    
    with open('Dataset/Molecule_Design/test.json','r',encoding='utf-8') as f:
        data = json.load(f)

    for i in tqdm(data[len_:]):
        smiles = i['SMILES']
        description = i['description']
        prompt = task_query + description
        start_time = time.time()
        history,all_tokens= await simulate(
            agents=agents,
            initial_prompt=prompt,
            mode=args.mode,
            max_steps=args.steps,
            no_tool=args.no_tool
        )
        references = ""
        for key, value_list in history.items():
            references= f"Agent{key}:\n"
            for index, value in enumerate(value_list):
                references+=f"Answer{index + 1}: {value}"
        
        refer_prompt = FinalRefer_prompt.format(question=prompt,answers=references)
        response,add_tokens= FinalRefer_node._run(refer_prompt)
        final_answer = response.split('answer is ')[-1].strip().strip("'")
        end_time = time.time()
        # response = ""
        # add_tokens = 0
        # final_answer = history[0][0].split('answer is ')[-1].strip().strip("'")
        new_data.append({
            'SMILES':smiles,
            'description':description,
            'answer':final_answer,
            'FinalRefer':response,
            'history':history,
            'all_tokens':all_tokens+add_tokens,
            'time':round(end_time - start_time, 3)
        })

        with open(filepath,'w',encoding='utf-8') as f:
            json.dump(new_data,f,ensure_ascii=False,indent=4)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())