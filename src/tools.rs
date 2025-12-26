//! Tool/function definition types for LLM providers.
//!
//! This module provides structures for defining tools that can be called by LLMs.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Definition of a tool that can be called by the LLM.
///
/// This is a re-export of the type from `types` module for convenience.
pub use crate::types::ToolDefinition;

/// Tool choice strategy for completion requests.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    /// Let the model decide whether to use tools
    #[default]
    Auto,
    /// Force the model to use a specific tool
    Tool { name: String },
    /// Force the model to use any tool
    Any,
    /// Prevent the model from using tools
    None,
}

impl ToolChoice {
    /// Create a tool choice that forces a specific tool.
    pub fn tool(name: impl Into<String>) -> Self {
        ToolChoice::Tool { name: name.into() }
    }
}

/// Builder for creating tool definitions.
#[derive(Debug, Clone)]
pub struct ToolBuilder {
    name: String,
    description: String,
    properties: serde_json::Map<String, Value>,
    required: Vec<String>,
}

impl ToolBuilder {
    /// Create a new tool builder with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            properties: serde_json::Map::new(),
            required: Vec::new(),
        }
    }

    /// Set the tool's description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Add a string parameter.
    pub fn string_param(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.insert(
            name.clone(),
            serde_json::json!({
                "type": "string",
                "description": description.into()
            }),
        );
        if required {
            self.required.push(name);
        }
        self
    }

    /// Add an integer parameter.
    pub fn integer_param(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.insert(
            name.clone(),
            serde_json::json!({
                "type": "integer",
                "description": description.into()
            }),
        );
        if required {
            self.required.push(name);
        }
        self
    }

    /// Add a number parameter.
    pub fn number_param(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.insert(
            name.clone(),
            serde_json::json!({
                "type": "number",
                "description": description.into()
            }),
        );
        if required {
            self.required.push(name);
        }
        self
    }

    /// Add a boolean parameter.
    pub fn boolean_param(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.insert(
            name.clone(),
            serde_json::json!({
                "type": "boolean",
                "description": description.into()
            }),
        );
        if required {
            self.required.push(name);
        }
        self
    }

    /// Add an array parameter.
    pub fn array_param(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        item_type: &str,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.insert(
            name.clone(),
            serde_json::json!({
                "type": "array",
                "description": description.into(),
                "items": { "type": item_type }
            }),
        );
        if required {
            self.required.push(name);
        }
        self
    }

    /// Add an enum parameter.
    pub fn enum_param(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        values: &[&str],
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.insert(
            name.clone(),
            serde_json::json!({
                "type": "string",
                "description": description.into(),
                "enum": values
            }),
        );
        if required {
            self.required.push(name);
        }
        self
    }

    /// Add a parameter with a custom JSON schema.
    pub fn custom_param(mut self, name: impl Into<String>, schema: Value, required: bool) -> Self {
        let name = name.into();
        self.properties.insert(name.clone(), schema);
        if required {
            self.required.push(name);
        }
        self
    }

    /// Build the tool definition.
    pub fn build(self) -> ToolDefinition {
        let input_schema = serde_json::json!({
            "type": "object",
            "properties": self.properties,
            "required": self.required
        });

        ToolDefinition {
            name: self.name,
            description: self.description,
            input_schema,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_builder() {
        let tool = ToolBuilder::new("bash")
            .description("Execute a shell command")
            .string_param("command", "The command to execute", true)
            .integer_param("timeout", "Timeout in seconds", false)
            .build();

        assert_eq!(tool.name, "bash");
        assert_eq!(tool.description, "Execute a shell command");

        let schema = &tool.input_schema;
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["command"]["type"] == "string");
        assert!(schema["required"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("command")));
    }

    #[test]
    fn test_enum_param() {
        let tool = ToolBuilder::new("select")
            .description("Select an option")
            .enum_param("option", "The option to select", &["a", "b", "c"], true)
            .build();

        let schema = &tool.input_schema;
        let option_schema = &schema["properties"]["option"];
        assert_eq!(option_schema["type"], "string");
        assert_eq!(option_schema["enum"], serde_json::json!(["a", "b", "c"]));
    }
}
