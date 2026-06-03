from pydantic import BaseModel

class AgentConfig(BaseModel):
    """Defines the configuration for an agent, including its name, description, capabilities, and constraints."""
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