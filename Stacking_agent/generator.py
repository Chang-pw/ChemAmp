from typing import Union, List
from .agent import Agent
import ast
from .tools import *
from .utils import * 

def to_list(input_string):
    try:
        result = ast.literal_eval(input_string)
        if isinstance(result, list):
            return result
    except (ValueError, SyntaxError):
        pass
    return [input_string]

class ToolGenerator:
    def __init__(self):
        self.tool_mapping = {
            'Name2SMILES': 'Name2SMILES()',
            'ChemDFM': 'ChemDFM()',
            'SMILES2Property':'SMILES2Property()',
            'SMILES2Description':'SMILES2Description()',
            'TextChemT5':'TextChemT5()',
            'UniMol':'UniMol()',
            'Llama':'Llama()',
            'Deepseek':'Deepseek()',
            'Chemformer':'Chemformer()',
        }

    def parse_tool_string(self, tool_str: str) -> List[str]:
        if '_' not in tool_str:
            return [tool_str, '0']
        return tool_str.split('_')

    def generate_single_tool(self, tool_spec: str) -> List[str]:
        tool_name, level = self.parse_tool_string(tool_spec)
        level = int(level)
        base_name = tool_name.lower()

        code_lines = []
        
        code_lines.append(f"{base_name}_0 = {self.tool_mapping[tool_name]}")

        for i in range(1, level + 1):
            deps = [f"{base_name}_{i-1}", f"{base_name}_0"]
            deps_str = ','.join(deps)
            code_lines.append(f"{base_name}_{i} = Agent_tool(Agent([{deps_str}]))")

        return code_lines

    def generate_combined_tools(self, tools: Union[List, str]) -> List[str]:
        if isinstance(tools, str):
            return self.generate_single_tool(tools)

        code_lines = []
        tool_outputs = []

        
        for tool in tools:
            if isinstance(tool, list):
                
                sub_lines = self.generate_combined_tools(tool)
                code_lines.extend(sub_lines)
                tool_outputs.append(sub_lines[-1].split(' = ')[0])
            else:
                sub_lines = self.generate_single_tool(tool)
                code_lines.extend(sub_lines)
                tool_outputs.append(sub_lines[-1].split(' = ')[0])

        if len(tool_outputs) > 1:
            combined_name = '__'.join(tool_outputs)
            deps_str = ','.join(tool_outputs)
            code_lines.append(f"final_agent = Agent_tool(Agent([{deps_str}]))")
        elif len(tool_outputs) == 1:
            code_lines.append(f"final_agent = {tool_outputs[0]}")

        return code_lines

    def generate(self, spec):
        spec = to_list(spec)
        code_lines = self.generate_combined_tools(spec)
        code= '\n'.join(code_lines)
        if code.count('\n') + 1 > 2:
            wo = False
        else:
            wo = True
        exec(code, globals())
        return final_agent,wo

       
def generate_tool(spec):
    pass

if __name__ == "__main__":
    generator = ToolGenerator()
    spec = "Name2SMILES_0"
    code = generator.generate(spec)._run