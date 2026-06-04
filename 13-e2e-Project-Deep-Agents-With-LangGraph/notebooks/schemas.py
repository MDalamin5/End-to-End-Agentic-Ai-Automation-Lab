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


class AgentAction(BaseModel):    
    """Defines an action taken by an agent, including the agent's name, the action description, and any relevant details."""
    agent_name: str
    action_description: str
    details: dict | None = None     


class AgentState(BaseModel):
    """Defines the state of an agent, including its name, current status, and any relevant information."""
    agent_name: str
    status: str  # e.g., "idle", "active", "completed"
    info: dict | None = None
    
class TaskProgress(BaseModel):
    """Defines the progress of a task, including the task name, current status, and any relevant details."""
    task_name: str
    status: str  # e.g., "not started", "in progress", "completed"
    details: dict | None = None

class TaskSummary(BaseModel):
    """Defines a summary of a task, including the task name, final status, and any relevant insights or conclusions."""
    task_name: str
    final_status: str  # e.g., "success", "failure"
    insights: str | None = None

class AgentInteraction(BaseModel):
    """Defines an interaction between agents, including the names of the agents involved, the interaction description, and any relevant details."""
    agent_1: str
    agent_2: str
    interaction_description: str
    details: dict | None = None

class TaskExecutionLog(BaseModel):
    """Defines a log entry for task execution, including the timestamp, task name, agent actions, and any relevant details."""
    timestamp: str  # ISO 8601 format
    task_name: str
    agent_actions: list[AgentAction]
    details: dict | None = None