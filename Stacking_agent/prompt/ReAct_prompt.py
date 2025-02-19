TOOL_DESC = """
{tool_name}: {tool_description} 
Parameters: {parameters} Format the arguments as a JSON object.
"""

REACT_PROMPT = """
### QUESTION
You are an expert chemist with critical thinking and your task is to respond to the question or solve the problem to the best of your ability.
Answer the following questions. You have access to the following tools. And you must call the following tools at least once:

{tool_descs}

Importantly: You should have your own reasoning logic and judgment. please note that the results returned by this tool may not necessarily be correct. You need to use them as a basis for inference along with the context I provide.

### FORMAT
Use the following format, the name of the chain (Thought,Action,Action Input,Observation,Final Answer) needs to be stated before each part:

Thought: reflect on your progress, think and deduct from your observation, then decide what to do next
Action: the action to take, should be one of [{tool_names}], Please output "Action: ..." .    
Action Input: the input to the action, should be formated into a valid JSON object.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated multiple times)

Thought: I now know the final answer
Final Answer: the final answer to the original input question, Noth that Please stated 'Final Answer:' before answering the question , NO OTHER INFORMATION or EXPAINATION!

### Begin! Utilize the results returned by this tool as much as possible!
"""

REACT_PROMPT_ = """
### QUESTION
You are an expert chemist with critical thinking and your task is to respond to the question or solve the problem to the best of your ability.
Answer the following questions. You have access to the following tools:

{tool_descs}

Importantly: You should have your own reasoning logic and judgment. please note that the results returned by this tool may not necessarily be correct. You need to use them as a basis for inference along with the context I provide.

### FORMAT
Use the following format, the name of the chain (Thought,Action,Action Input,Observation,Final Answer) needs to be stated before each part:

Thought: reflect on your progress, think and deduct from your observation, then decide what to do next
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, should be formated into a valid JSON object.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated multiple times)

Thought: I now know the final answer
Final Answer: the final answer to the original input question, Noth that Please stated 'Final Answer:' before answering the question , NO OTHER INFORMATION or EXPAINATION!

### Begin!
"""
