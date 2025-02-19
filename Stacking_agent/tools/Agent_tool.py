import threading

class Agent_tool:
    instance_count = 0
    instance_lock = threading.Lock()  
    description = ''
    task_name = ''
    task_query = ''

    def __init__(self,agent,data=[],description='',next_n=True,debug=False,data_index=0,**tool_args):
        with Agent_tool.instance_lock:  # 加锁，确保线程安全
            if next_n:
                Agent_tool.instance_count += 1
            self.number = Agent_tool.instance_count
        self.agent = agent
        self.data = data
        self.name = f'{Agent_tool.task_name}_{self.number}'
        self.debug = debug
        self.index = data_index
        self.task_query = Agent_tool.task_query
    def _run(self,query:str,**tool_args)->str:
        return  self.data[self.index]["answer"]
    def test_run(self,query:str,debug=True,**tool_args)->str:
        final,response,hs,all_tokens = self.agent._run(query,[],debug=debug,test=True)
        return final,all_tokens
    
    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    @classmethod
    def set_description(cls, new_description: str):
        cls.description = new_description

    @classmethod
    def set_task_name(cls, new_task_name: str):
        cls.task_name = new_task_name
    
    @classmethod
    def set_task_query(cls, new_task_query: str):
        cls.task_query = new_task_query
    