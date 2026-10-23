from model import model
from deepagents import create_deep_agent

SYSTEM_PROMPT = "You are a salty pirate captain"

agent = create_deep_agent(
    model= model,
    system_prompt=SYSTEM_PROMPT,
    name="pagla_gora"
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is an LLM!"
            }
        ]
    }
)

print(result["messages"][-1].text)