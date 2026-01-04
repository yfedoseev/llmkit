"""
Structured Output (JSON Schema) Example

Demonstrates how to get structured JSON responses using schema enforcement.

Requirements:
- Set OPENAI_API_KEY environment variable

Run:
    python 05_structured_output.py
"""

import json
from modelsuite import LLMKitClient, Message, CompletionRequest


def simple_json_output():
    """Get JSON output without a specific schema."""
    client = LLMKitClient.from_env()

    # Use "provider/model" format for explicit provider routing
    request = CompletionRequest(
        model="openai/gpt-4o",
        messages=[
            Message.user(
                "Generate a JSON object representing a fictional book with "
                "title, author, year, and genre fields."
            )
        ],
        max_tokens=200,
    ).with_json_output()  # Enables JSON mode

    print("Getting simple JSON output...")
    response = client.complete(request)

    try:
        data = json.loads(response.text_content())
        print(f"\nParsed JSON:\n{json.dumps(data, indent=2)}")
    except json.JSONDecodeError:
        print(f"\nRaw response:\n{response.text_content()}")


def schema_enforced_output():
    """Get JSON output conforming to a specific schema."""
    client = LLMKitClient.from_env()

    # Define the expected JSON schema
    person_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Full name of the person",
            },
            "age": {
                "type": "integer",
                "description": "Age in years",
                "minimum": 0,
                "maximum": 150,
            },
            "email": {
                "type": "string",
                "description": "Email address",
            },
            "occupation": {
                "type": "string",
                "description": "Current job title",
            },
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of skills",
            },
        },
        "required": ["name", "age", "email", "occupation"],
    }

    request = CompletionRequest(
        model="openai/gpt-4o",
        messages=[
            Message.user(
                "Generate a fictional software engineer's profile. "
                "Make it realistic with appropriate skills."
            )
        ],
        max_tokens=500,
    ).with_json_schema("person_profile", person_schema)

    print("Getting schema-enforced JSON output...")
    response = client.complete(request)

    data = json.loads(response.text_content())
    print(f"\nPerson profile:\n{json.dumps(data, indent=2)}")

    # Validate the response
    print("\nValidation:")
    print(f"  Name: {data.get('name', 'MISSING')}")
    print(f"  Age: {data.get('age', 'MISSING')}")
    print(f"  Email: {data.get('email', 'MISSING')}")
    print(f"  Occupation: {data.get('occupation', 'MISSING')}")
    if "skills" in data:
        print(f"  Skills: {', '.join(data['skills'])}")


def complex_nested_schema():
    """Example with a more complex nested schema."""
    client = LLMKitClient.from_env()

    # Schema for a product catalog entry
    product_schema = {
        "type": "object",
        "properties": {
            "product": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "price": {"type": "number"},
                    "currency": {"type": "string", "enum": ["USD", "EUR", "GBP"]},
                },
                "required": ["id", "name", "price", "currency"],
            },
            "inventory": {
                "type": "object",
                "properties": {
                    "quantity": {"type": "integer"},
                    "warehouse": {"type": "string"},
                    "last_updated": {"type": "string"},
                },
                "required": ["quantity", "warehouse"],
            },
            "categories": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["product", "inventory"],
    }

    request = CompletionRequest(
        model="openai/gpt-4o",
        messages=[
            Message.user(
                "Generate a product entry for a fictional electronics item."
            )
        ],
        max_tokens=500,
    ).with_json_schema("product_entry", product_schema)

    print("Getting complex nested JSON...")
    response = client.complete(request)

    data = json.loads(response.text_content())
    print(f"\nProduct entry:\n{json.dumps(data, indent=2)}")


def list_extraction():
    """Extract a list of items from text."""
    client = LLMKitClient.from_env()

    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "quantity": {"type": "integer"},
                        "category": {
                            "type": "string",
                            "enum": ["produce", "dairy", "meat", "pantry", "other"],
                        },
                    },
                    "required": ["name", "quantity", "category"],
                },
            },
        },
        "required": ["items"],
    }

    text = """
    I need to buy some groceries:
    - 2 pounds of chicken breast
    - A dozen eggs
    - Milk (1 gallon)
    - 5 apples
    - Rice (2 bags)
    - Cheddar cheese
    """

    request = CompletionRequest(
        model="openai/gpt-4o",
        messages=[
            Message.user(f"Extract the shopping list from this text:\n{text}")
        ],
        max_tokens=500,
    ).with_json_schema("shopping_list", schema)

    print("Extracting structured list from text...")
    response = client.complete(request)

    data = json.loads(response.text_content())
    print("\nExtracted shopping list:")
    for item in data.get("items", []):
        print(f"  - {item['name']}: {item['quantity']} ({item['category']})")


def main():
    print("=" * 50)
    print("Example 1: Simple JSON Output")
    print("=" * 50)
    simple_json_output()

    print("\n" + "=" * 50)
    print("Example 2: Schema-Enforced Output")
    print("=" * 50)
    schema_enforced_output()

    print("\n" + "=" * 50)
    print("Example 3: Complex Nested Schema")
    print("=" * 50)
    complex_nested_schema()

    print("\n" + "=" * 50)
    print("Example 4: List Extraction")
    print("=" * 50)
    list_extraction()


if __name__ == "__main__":
    main()
