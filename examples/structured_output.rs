//! Structured Output (JSON Schema) Example
//!
//! Demonstrates how to get structured JSON responses using schema enforcement.
//!
//! Requirements:
//! - Set OPENAI_API_KEY environment variable
//!
//! Run with:
//!     cargo run --example structured_output --features openai

use modelsuite::{CompletionRequest, Message, ModelSuiteClient};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Serialize, Deserialize)]
struct Person {
    name: String,
    age: u32,
    email: String,
    occupation: String,
    #[serde(default)]
    skills: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Product {
    product: ProductInfo,
    inventory: InventoryInfo,
    #[serde(default)]
    categories: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ProductInfo {
    id: String,
    name: String,
    price: f64,
    currency: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct InventoryInfo {
    quantity: i32,
    warehouse: String,
    #[serde(default)]
    last_updated: Option<String>,
}

#[tokio::main]
async fn main() -> modelsuite::Result<()> {
    let client = ModelSuiteClient::builder()
        .with_openai_from_env()
        .with_default_retry()
        .build()
        .await?;

    // Example 1: Simple JSON output
    println!("{}", "=".repeat(50));
    println!("Example 1: Simple JSON Output");
    println!("{}", "=".repeat(50));
    simple_json_output(&client).await?;

    // Example 2: Schema-enforced output
    println!("\n{}", "=".repeat(50));
    println!("Example 2: Schema-Enforced Output");
    println!("{}", "=".repeat(50));
    schema_enforced_output(&client).await?;

    // Example 3: Complex nested schema
    println!("\n{}", "=".repeat(50));
    println!("Example 3: Complex Nested Schema");
    println!("{}", "=".repeat(50));
    complex_nested_schema(&client).await?;

    Ok(())
}

async fn simple_json_output(client: &ModelSuiteClient) -> modelsuite::Result<()> {
    // Use "provider/model" format for explicit provider routing
    let request = CompletionRequest::new(
        "openai/gpt-4o",
        vec![Message::user(
            "Generate a JSON object representing a fictional book with \
             title, author, year, and genre fields. Return ONLY valid JSON.",
        )],
    )
    .with_max_tokens(200)
    .with_json_output();

    println!("Getting simple JSON output...");
    let response = client.complete(request).await?;

    match serde_json::from_str::<serde_json::Value>(&response.text_content()) {
        Ok(data) => {
            println!("\nParsed JSON:");
            println!("{}", serde_json::to_string_pretty(&data).unwrap());
        }
        Err(_) => {
            println!("\nRaw response:\n{}", response.text_content());
        }
    }

    Ok(())
}

async fn schema_enforced_output(client: &ModelSuiteClient) -> modelsuite::Result<()> {
    // Define the expected JSON schema
    let person_schema = json!({
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Full name of the person"
            },
            "age": {
                "type": "integer",
                "description": "Age in years",
                "minimum": 0,
                "maximum": 150
            },
            "email": {
                "type": "string",
                "description": "Email address"
            },
            "occupation": {
                "type": "string",
                "description": "Current job title"
            },
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of skills"
            }
        },
        "required": ["name", "age", "email", "occupation"]
    });

    let request = CompletionRequest::new(
        "openai/gpt-4o",
        vec![Message::user(
            "Generate a fictional software engineer's profile. \
             Make it realistic with appropriate skills.",
        )],
    )
    .with_max_tokens(500)
    .with_json_schema("person_profile", person_schema);

    println!("Getting schema-enforced JSON output...");
    let response = client.complete(request).await?;

    let person: Person = serde_json::from_str(&response.text_content())?;
    println!("\nPerson profile:");
    println!("{}", serde_json::to_string_pretty(&person).unwrap());

    println!("\nValidation:");
    println!("  Name: {}", person.name);
    println!("  Age: {}", person.age);
    println!("  Email: {}", person.email);
    println!("  Occupation: {}", person.occupation);
    if !person.skills.is_empty() {
        println!("  Skills: {}", person.skills.join(", "));
    }

    Ok(())
}

async fn complex_nested_schema(client: &ModelSuiteClient) -> modelsuite::Result<()> {
    // Schema for a product catalog entry
    let product_schema = json!({
        "type": "object",
        "properties": {
            "product": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "price": {"type": "number"},
                    "currency": {"type": "string", "enum": ["USD", "EUR", "GBP"]}
                },
                "required": ["id", "name", "price", "currency"]
            },
            "inventory": {
                "type": "object",
                "properties": {
                    "quantity": {"type": "integer"},
                    "warehouse": {"type": "string"},
                    "last_updated": {"type": "string"}
                },
                "required": ["quantity", "warehouse"]
            },
            "categories": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["product", "inventory"]
    });

    let request = CompletionRequest::new(
        "openai/gpt-4o",
        vec![Message::user(
            "Generate a product entry for a fictional electronics item.",
        )],
    )
    .with_max_tokens(500)
    .with_json_schema("product_entry", product_schema);

    println!("Getting complex nested JSON...");
    let response = client.complete(request).await?;

    let product: Product = serde_json::from_str(&response.text_content())?;
    println!("\nProduct entry:");
    println!("{}", serde_json::to_string_pretty(&product).unwrap());

    Ok(())
}
