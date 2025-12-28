//! Tool definition types for JavaScript bindings

use llmkit::tools::ToolBuilder;
use llmkit::types::ToolDefinition;
use napi_derive::napi;

/// Definition of a tool that can be used by the model.
#[napi]
#[derive(Clone)]
pub struct JsToolDefinition {
    pub(crate) inner: ToolDefinition,
}

#[napi]
impl JsToolDefinition {
    /// Create a new tool definition.
    #[napi(constructor)]
    pub fn new(name: String, description: String, input_schema: serde_json::Value) -> Self {
        Self {
            inner: ToolDefinition::new(name, description, input_schema),
        }
    }

    /// The tool name.
    #[napi(getter)]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// The tool description.
    #[napi(getter)]
    pub fn description(&self) -> String {
        self.inner.description.clone()
    }

    /// The input schema.
    #[napi(getter)]
    pub fn input_schema(&self) -> serde_json::Value {
        self.inner.input_schema.clone()
    }
}

impl From<ToolDefinition> for JsToolDefinition {
    fn from(tool: ToolDefinition) -> Self {
        Self { inner: tool }
    }
}

impl From<JsToolDefinition> for ToolDefinition {
    fn from(js_tool: JsToolDefinition) -> Self {
        js_tool.inner
    }
}

/// Builder for creating tool definitions with a fluent API.
///
/// @example
/// ```typescript
/// const tool = new ToolBuilder("get_weather")
///   .description("Get current weather")
///   .stringParam("city", "City name", true)
///   .enumParam("unit", "Temperature unit", ["celsius", "fahrenheit"])
///   .build()
/// ```
#[napi]
#[derive(Clone)]
pub struct JsToolBuilder {
    inner: ToolBuilder,
}

#[napi]
impl JsToolBuilder {
    /// Create a new tool builder.
    #[napi(constructor)]
    pub fn new(name: String) -> Self {
        Self {
            inner: ToolBuilder::new(name),
        }
    }

    /// Set the tool description.
    #[napi]
    pub fn description(&self, description: String) -> Self {
        Self {
            inner: self.inner.clone().description(description),
        }
    }

    /// Add a string parameter.
    #[napi]
    pub fn string_param(&self, name: String, description: String, required: Option<bool>) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .string_param(name, description, required.unwrap_or(true)),
        }
    }

    /// Add an integer parameter.
    #[napi]
    pub fn integer_param(&self, name: String, description: String, required: Option<bool>) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .integer_param(name, description, required.unwrap_or(true)),
        }
    }

    /// Add a number (float) parameter.
    #[napi]
    pub fn number_param(&self, name: String, description: String, required: Option<bool>) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .number_param(name, description, required.unwrap_or(true)),
        }
    }

    /// Add a boolean parameter.
    #[napi]
    pub fn boolean_param(&self, name: String, description: String, required: Option<bool>) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .boolean_param(name, description, required.unwrap_or(true)),
        }
    }

    /// Add an array parameter.
    #[napi]
    pub fn array_param(
        &self,
        name: String,
        description: String,
        item_type: String,
        required: Option<bool>,
    ) -> Self {
        Self {
            inner: self.inner.clone().array_param(
                name,
                description,
                &item_type,
                required.unwrap_or(true),
            ),
        }
    }

    /// Add an enum parameter (string with allowed values).
    #[napi]
    pub fn enum_param(
        &self,
        name: String,
        description: String,
        values: Vec<String>,
        required: Option<bool>,
    ) -> Self {
        let values_ref: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        Self {
            inner: self.inner.clone().enum_param(
                name,
                description,
                &values_ref,
                required.unwrap_or(true),
            ),
        }
    }

    /// Add a custom parameter with a JSON schema.
    #[napi]
    pub fn custom_param(
        &self,
        name: String,
        schema: serde_json::Value,
        required: Option<bool>,
    ) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .custom_param(name, schema, required.unwrap_or(true)),
        }
    }

    /// Build the tool definition.
    #[napi]
    pub fn build(&self) -> JsToolDefinition {
        JsToolDefinition {
            inner: self.inner.clone().build(),
        }
    }
}
