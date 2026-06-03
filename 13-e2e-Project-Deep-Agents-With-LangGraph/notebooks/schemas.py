from pydantic import BaseModel

class AgentConfig(BaseModel):
    name: str
    description: str
    capabilities: list[str]
    constraints: list[str]  

class TaskConfig(BaseModel):
    """Defines the configuration for a task, including its name, description, and the agents involved.
    """
    task_name: str
    task_description: str
    agents: list[AgentConfig]