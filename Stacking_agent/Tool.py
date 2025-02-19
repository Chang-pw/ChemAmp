from .utils import *
from .tools import *

class Tools:
    def __init__(self, all_tool: list) -> None:
        self.toolConfig = self._tools(all_tool)
        self.initConfig = self._inittools(all_tool)
    def _tools(self, all_tool: list) -> list:
        tool_list = []
        for tool_class in all_tool:
            tool_instance = tool_class
            tool_name = tool_instance.name
            tool_description = tool_instance.description
            tool_config = function_to_json(tool_instance._run)
            tool_config["tool_name"] = tool_name
            tool_config["tool_description"] = tool_description
            tool_list.append(tool_config)
        return tool_list
    def _inittools(self,all_tool: list) -> list:
        tool_list = []
        for tool_class in all_tool:
            dict = extract_instance_params(tool_class)
            dict['tool_name'] = tool_class.name
            tool_list.append(dict)
        return tool_list
    
    def __call__(self, tool_name: str,data_index=0,test=False,**tool_args):
        for tool in self.toolConfig:
            if tool["tool_name"] == tool_name:
                tool_names = ("Molecule_Design", "Molecule_captioning", "MolecularPropertyPrediction","ReactionPrediction")
                if any(name in tool_name for name in tool_names):
                    tool_name = 'Agent_tool'
                    tool_args['next_n'] = False
                    tool_args['data_index'] = data_index
                    if test:
                        tool_class = globals()[tool_name]
                        tool_instance = tool_class(**tool_args)  
                        result,tokens = tool_instance.test_run(**tool_args)
                        return result,tokens
                tool_class = globals()[tool_name]
                tool_instance = tool_class(**tool_args)
                return tool_instance._run(**tool_args),0
        return "Tool not found, please only input the tool name",0

