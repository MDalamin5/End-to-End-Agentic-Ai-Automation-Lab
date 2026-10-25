from langchain_core.tools import tool


@tool
def getSum(a: int, b: int) -> int:
    """This function return two int number sum."""

    return a + b