"""
Tool Calling (Function Calling) Example

Demonstrates how to define and use tools with LLMKit.

Requirements:
- Set ANTHROPIC_API_KEY environment variable (or another provider's key)

Run:
    python 03_tool_calling.py
"""

import json
from modelsuite import (
    LLMKitClient,
    Message,
    CompletionRequest,
    ToolBuilder,
    ContentBlock,
)


def get_weather(city: str, unit: str = "celsius") -> dict:
    """Simulated weather function."""
    # In a real app, this would call a weather API
    weather_data = {
        "paris": {"temp": 22, "condition": "sunny"},
        "london": {"temp": 15, "condition": "cloudy"},
        "tokyo": {"temp": 28, "condition": "humid"},
    }

    data = weather_data.get(city.lower(), {"temp": 20, "condition": "unknown"})

    if unit == "fahrenheit":
        data["temp"] = data["temp"] * 9 // 5 + 32

    return {
        "city": city,
        "temperature": data["temp"],
        "unit": unit,
        "condition": data["condition"],
    }


def main():
    client = LLMKitClient.from_env()

    # Define a tool using the builder pattern
    weather_tool = (
        ToolBuilder("get_weather")
        .description("Get the current weather for a city")
        .string_param("city", "The city name to get weather for", required=True)
        .enum_param(
            "unit",
            "Temperature unit",
            ["celsius", "fahrenheit"],
            required=False,
        )
        .build()
    )

    # Initial request with tools
    # Use "provider/model" format for explicit provider routing
    request = CompletionRequest(
        model="anthropic/claude-sonnet-4-20250514",
        messages=[Message.user("What's the weather like in Paris today?")],
        max_tokens=1024,
    ).with_tools([weather_tool])

    print("Sending initial request with tools...")
    response = client.complete(request)

    # Check if the model wants to use a tool
    if response.has_tool_use():
        print("\nModel wants to use tools:")

        for tool_use in response.tool_uses():
            tool_info = tool_use.as_tool_use()
            if tool_info:
                tool_id, tool_name, tool_input = tool_info

                print(f"  Tool: {tool_name}")
                print(f"  Input: {tool_input}")

                # Execute the tool
                if tool_name == "get_weather":
                    result = get_weather(
                        city=tool_input.get("city", ""),
                        unit=tool_input.get("unit", "celsius"),
                    )
                    result_json = json.dumps(result)
                    print(f"  Result: {result_json}")

                    # Create tool result
                    tool_result = ContentBlock.tool_result(
                        tool_use_id=tool_id,
                        content=result_json,
                        is_error=False,
                    )

                    # Continue conversation with tool results
                    messages = [
                        Message.user("What's the weather like in Paris today?"),
                        Message.assistant_with_content(response.content),
                        Message.tool_results([tool_result]),
                    ]

                    print("\nSending tool results back to model...")
                    final_response = client.complete(
                        CompletionRequest(
                            model="anthropic/claude-sonnet-4-20250514",
                            messages=messages,
                            max_tokens=1024,
                        ).with_tools([weather_tool])
                    )

                    print(f"\nFinal response: {final_response.text_content()}")
    else:
        print(f"\nResponse: {response.text_content()}")


def multi_tool_example():
    """Example with multiple tools."""
    client = LLMKitClient.from_env()

    # Define multiple tools
    weather_tool = (
        ToolBuilder("get_weather")
        .description("Get current weather")
        .string_param("city", "City name")
        .build()
    )

    time_tool = (
        ToolBuilder("get_time")
        .description("Get current time in a timezone")
        .string_param("timezone", "Timezone name (e.g., America/New_York)")
        .build()
    )

    calculator_tool = (
        ToolBuilder("calculate")
        .description("Perform mathematical calculations")
        .string_param("expression", "Mathematical expression to evaluate")
        .build()
    )

    request = CompletionRequest(
        model="anthropic/claude-sonnet-4-20250514",
        messages=[
            Message.user("What's 25 * 4? Also what's the weather in Tokyo?")
        ],
        max_tokens=1024,
    ).with_tools([weather_tool, time_tool, calculator_tool])

    response = client.complete(request)

    print("\nMulti-tool response:")
    print(f"Stop reason: {response.stop_reason}")

    if response.has_tool_use():
        for tool_use in response.tool_uses():
            tool_info = tool_use.as_tool_use()
            if tool_info:
                print(f"  Tool requested: {tool_info[1]} with {tool_info[2]}")


if __name__ == "__main__":
    main()
    # Uncomment for multi-tool example:
    # multi_tool_example()
