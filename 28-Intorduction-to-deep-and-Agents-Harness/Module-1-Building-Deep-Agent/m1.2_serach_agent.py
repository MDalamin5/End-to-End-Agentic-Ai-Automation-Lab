from model import model
from deepagents import create_deep_agent
from langchain_core.tools import tool


@tool
def getSum(a: int, b: int) -> int:
    """This function return two int number sum."""

    return a + b

SYSTEM_PROMPT = "You are a salty pirate captain"



agent = create_deep_agent(
    model= model,
    system_prompt=SYSTEM_PROMPT,
    name="pagla_gora",
    tools= [getSum]
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the sum of 5 and 9"
            }
        ]
    }
)

print(result["messages"][-1].text)