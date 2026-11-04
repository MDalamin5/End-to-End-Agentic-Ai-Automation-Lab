from langchain.tools import tool


@tool
def getHotWeather(city: str):
    """THis is a tools use for any city weather is hot or not."""
    response = "warm 28 degree"
    if city =="DHK":
        response= "Hot 39 degree"

    return response