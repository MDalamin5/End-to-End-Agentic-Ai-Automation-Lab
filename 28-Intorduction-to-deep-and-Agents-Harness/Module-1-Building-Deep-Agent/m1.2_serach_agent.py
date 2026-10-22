from model import model
from deepagents import create_deep_agent

agent = create_deep_agent(model= model)

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