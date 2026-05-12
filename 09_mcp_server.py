"""
What it does: An MCP server with three tools (calculator, weather, converter).
"""

from fastmcp import FastMCP
import httpx

# Initialize the server. The name is used by the client.
mcp = FastMCP("ToolServer")


@mcp.tool()
def calculator(expression: str) -> str:
    """Calculates a mathematical expression.

    Args:
        expression (str): Mathematical expression as a string (e.g., '2 + 2' or '17 ** 5').

    Returns:
        str: Result of the calculation or an error message.
    """
    # try-except is necessary to prevent the server from crashing due to LLM generation errors
    try:
        # Note: Using eval is generally unsafe in production.
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
async def get_weather(city: str) -> str:
    """Returns the current weather in a specified city.

    Args:
        city (str): The name of the city.

    Returns:
        str: A string describing the current weather or an error notification.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://wttr.in/{city}?format=%C+%t",
            headers={"User-Agent": "AI-Engineer-Course/1.0"},
            timeout=5.0,
        )
        if response.status_code == 200:
            return f"Weather in {city}: {response.text.strip()}"
        return f"Failed to retrieve weather for {city}"


@mcp.tool()
def temperature_converter(celsius: float) -> str:
    """Converts temperature from Celsius to Fahrenheit and Kelvin.

    Args:
        celsius (float): Temperature in degrees Celsius.

    Returns:
        str: A string with the converted temperature values.
    """
    fahrenheit = celsius * 9 / 5 + 32
    kelvin = celsius + 273.15
    return f"{celsius}°C = {fahrenheit:.1f}°F = {kelvin:.2f}K"


if __name__ == "__main__":
    # transport="stdio" means communication via standard input/output (child process).
    # For remote servers, you can use "sse".
    mcp.run(transport="stdio")
