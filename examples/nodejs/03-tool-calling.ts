/**
 * Tool Calling (Function Calling) Example
 *
 * Demonstrates how to define and use tools with LLMKit.
 *
 * Requirements:
 * - Set ANTHROPIC_API_KEY environment variable (or another provider's key)
 *
 * Run:
 *   npx ts-node 03-tool-calling.ts
 */

import {
    JsLlmKitClient as LLMKitClient,
    JsMessage as Message,
    JsCompletionRequest as CompletionRequest,
    JsToolBuilder as ToolBuilder,
    JsContentBlock as ContentBlock,
} from 'modelsuite'

// Simulated weather function
function getWeather(city: string, unit: string = 'celsius'): object {
    const weatherData: Record<string, { temp: number; condition: string }> = {
        paris: { temp: 22, condition: 'sunny' },
        london: { temp: 15, condition: 'cloudy' },
        tokyo: { temp: 28, condition: 'humid' },
    }

    const data = weatherData[city.toLowerCase()] ?? { temp: 20, condition: 'unknown' }

    let temp = data.temp
    if (unit === 'fahrenheit') {
        temp = Math.floor(temp * 9 / 5 + 32)
    }

    return {
        city,
        temperature: temp,
        unit,
        condition: data.condition,
    }
}

async function main() {
    const client = LLMKitClient.fromEnv()

    // Define a tool using the builder pattern
    const weatherTool = new ToolBuilder('get_weather')
        .description('Get the current weather for a city')
        .stringParam('city', 'The city name to get weather for', true)
        .enumParam('unit', 'Temperature unit', ['celsius', 'fahrenheit'])
        .build()

    // Initial request with tools
    // Use "provider/model" format for explicit provider routing
    const request = CompletionRequest
        .create('anthropic/claude-sonnet-4-20250514', [
            Message.user("What's the weather like in Paris today?")
        ])
        .withMaxTokens(1024)
        .withTools([weatherTool])

    console.log('Sending initial request with tools...')
    const response = await client.complete(request)

    // Check if the model wants to use a tool
    if (response.hasToolUse()) {
        console.log('\nModel wants to use tools:')

        for (const toolUse of response.toolUses()) {
            const toolInfo = toolUse.asToolUse()
            if (toolInfo) {
                console.log('  Tool:', toolInfo.name)
                console.log('  Input:', JSON.stringify(toolInfo.input))

                // Execute the tool
                if (toolInfo.name === 'get_weather') {
                    const result = getWeather(
                        toolInfo.input.city ?? '',
                        toolInfo.input.unit ?? 'celsius'
                    )
                    const resultJson = JSON.stringify(result)
                    console.log('  Result:', resultJson)

                    // Create tool result
                    const toolResult = ContentBlock.toolResult(
                        toolInfo.id,
                        resultJson,
                        false
                    )

                    // Continue conversation with tool results
                    const messages = [
                        Message.user("What's the weather like in Paris today?"),
                        Message.assistantWithContent(response.content),
                        Message.toolResults([toolResult]),
                    ]

                    console.log('\nSending tool results back to model...')
                    const finalResponse = await client.complete(
                        CompletionRequest
                            .create('anthropic/claude-sonnet-4-20250514', messages)
                            .withMaxTokens(1024)
                            .withTools([weatherTool])
                    )

                    console.log('\nFinal response:', finalResponse.textContent())
                }
            }
        }
    } else {
        console.log('\nResponse:', response.textContent())
    }
}

async function multiToolExample() {
    const client = LLMKitClient.fromEnv()

    // Define multiple tools
    const weatherTool = new ToolBuilder('get_weather')
        .description('Get current weather')
        .stringParam('city', 'City name')
        .build()

    const timeTool = new ToolBuilder('get_time')
        .description('Get current time in a timezone')
        .stringParam('timezone', 'Timezone name (e.g., America/New_York)')
        .build()

    const calculatorTool = new ToolBuilder('calculate')
        .description('Perform mathematical calculations')
        .stringParam('expression', 'Mathematical expression to evaluate')
        .build()

    const request = CompletionRequest
        .create('anthropic/claude-sonnet-4-20250514', [
            Message.user("What's 25 * 4? Also what's the weather in Tokyo?")
        ])
        .withMaxTokens(1024)
        .withTools([weatherTool, timeTool, calculatorTool])

    const response = await client.complete(request)

    console.log('\nMulti-tool response:')
    console.log('Stop reason:', response.stopReason)

    if (response.hasToolUse()) {
        for (const toolUse of response.toolUses()) {
            const info = toolUse.asToolUse()
            if (info) {
                console.log(`  Tool requested: ${info.name} with`, info.input)
            }
        }
    }
}

main().catch(console.error)
