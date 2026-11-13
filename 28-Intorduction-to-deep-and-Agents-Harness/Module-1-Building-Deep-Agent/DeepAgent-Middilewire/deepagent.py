from deepagents import create_deep_agent
from model import model

SYSTEM_PROMPT = "Your are professional Thinker and always give Negative response"

agent = create_deep_agent(
    model=model,
    name="SQL_Agent",
    #tools=[read_sql, getHotWeather],
    system_prompt=SYSTEM_PROMPT,
)