from pydantic import BaseModel

class AgentConfig(BaseModel):
    """Defines the configuration for an agent, including its name, description, capabilities, and constraints."""
    name: str
    description: str
    capabilities: list[str]
    constraints: list[str]  

class TaskConfig(BaseModel):
    """Defines the configuration for a task, including its name, description, and the agents involved."""
    task_name: str
    task_description: str
    agents: list[AgentConfig]

class TaskExecutionResult(BaseModel):
    """Defines the result of executing a task, including the task name, status, and any output or error messages."""
    task_name: str
    status: str  # e.g., "success", "failure"
    output: str | None = None
    error_message: str | None = None

