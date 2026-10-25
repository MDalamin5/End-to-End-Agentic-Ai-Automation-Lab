from langchain_core.tools import tool


@tool
def getSum(a: int, b: int) -> int:
    """This function return two int number sum."""

    return a + b


@tool
def getPrompt(currentPrompt: str) -> str:
    """its update the system prompts as Natural."""

    return "You are natural message response provider."