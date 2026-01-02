//! Tool Calling (Function Calling) Example
//!
//! Demonstrates how to define and use tools with LLMKit.
//!
//! Requirements:
//! - Set ANTHROPIC_API_KEY environment variable (or another provider's key)
//!
//! Run with:
//!     cargo run --example tool_calling

use llmkit::{CompletionRequest, ContentBlock, LLMKitClient, Message, ToolDefinition};
use serde_json::json;

// Simulated weather function
fn get_weather(city: &str, unit: &str) -> serde_json::Value {
    let (temp, condition) = match city.to_lowercase().as_str() {
        "paris" => (22, "sunny"),
        "london" => (15, "cloudy"),
        "tokyo" => (28, "humid"),
        _ => (20, "unknown"),
    };

    let temp = if unit == "fahrenheit" {
        temp * 9 / 5 + 32
    } else {
        temp
    };

    json!({
        "city": city,
        "temperature": temp,
        "unit": unit,
        "condition": condition,
    })
}

#[tokio::main]
async fn main() -> llmkit::Result<()> {
    let client = LLMKitClient::builder()
        .with_anthropic_from_env()
        .with_default_retry()
        .build()?;

    // Define a tool
    let weather_tool = ToolDefinition {
        name: "get_weather".to_string(),
        description: "Get the current weather for a city".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name to get weather for"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["city"]
        }),
    };

    // Initial request with tools
    // Use "provider/model" format for explicit provider routing
    let request = CompletionRequest::new(
        "anthropic/claude-sonnet-4-20250514",
        vec![Message::user("What's the weather like in Paris today?")],
    )
    .with_max_tokens(1024)
    .with_tools(vec![weather_tool.clone()]);

    println!("Sending initial request with tools...");
    let response = client.complete(request).await?;

    // Check if the model wants to use a tool
    if response.has_tool_use() {
        println!("\nModel wants to use tools:");

        for content in &response.content {
            if let ContentBlock::ToolUse { id, name, input } = content {
                println!("  Tool: {}", name);
                println!("  Input: {}", input);

                // Execute the tool
                if name == "get_weather" {
                    let city = input
                        .get("city")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    let unit = input
                        .get("unit")
                        .and_then(|v| v.as_str())
                        .unwrap_or("celsius");

                    let result = get_weather(city, unit);
                    let result_json = result.to_string();
                    println!("  Result: {}", result_json);

                    // Create tool result
                    let tool_result = ContentBlock::ToolResult {
                        tool_use_id: id.clone(),
                        content: result_json,
                        is_error: false,
                    };

                    // Continue conversation with tool results
                    let messages = vec![
                        Message::user("What's the weather like in Paris today?"),
                        Message::assistant_with_content(response.content.clone()),
                        Message::user_with_content(vec![tool_result]),
                    ];

                    println!("\nSending tool results back to model...");
                    let final_response = client
                        .complete(
                            CompletionRequest::new("anthropic/claude-sonnet-4-20250514", messages)
                                .with_max_tokens(1024)
                                .with_tools(vec![weather_tool.clone()]),
                        )
                        .await?;

                    println!("\nFinal response: {}", final_response.text_content());
                }
            }
        }
    } else {
        println!("\nResponse: {}", response.text_content());
    }

    Ok(())
}
