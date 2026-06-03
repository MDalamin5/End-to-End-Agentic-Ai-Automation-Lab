from pydantic import BaseModel

class AgentConfig(BaseModel):
    name: str
    description: str
    capabilities: list[str]
    constraints: list[str]  

class TaskConfig(BaseModel):
    task_name: str
    task_description: str
    agents: list[AgentConfig]