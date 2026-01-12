//! Tool definition types for Python bindings

use llmkit::tools::ToolBuilder;
use llmkit::types::ToolDefinition;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Definition of a tool that can be used by the model.
#[pyclass(name = "ToolDefinition")]
#[derive(Clone)]
pub struct PyToolDefinition {
    pub(crate) inner: ToolDefinition,
}

#[pymethods]
impl PyToolDefinition {
    /// Create a new tool definition.
    ///
    /// Args:
    ///     name: Tool name
    ///     description: Tool description
    ///     input_schema: JSON Schema for the tool's input
    ///
    /// Returns:
    ///     ToolDefinition: A new tool definition
    #[new]
    fn new(name: String, description: String, input_schema: Bound<'_, PyDict>) -> PyResult<Self> {
        let schema: serde_json::Value = pythonize::depythonize(&input_schema)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: ToolDefinition::new(name, description, schema),
        })
    }

    /// The tool name.
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// The tool description.
    #[getter]
    fn description(&self) -> &str {
        &self.inner.description
    }

    /// The input schema as a dictionary.
    #[getter]
    fn input_schema(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        pythonize::pythonize(py, &self.inner.input_schema)
            .map(|obj| obj.into())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "ToolDefinition(name={:?}, description={:?})",
            self.inner.name, self.inner.description
        )
    }
}

impl From<ToolDefinition> for PyToolDefinition {
    fn from(tool: ToolDefinition) -> Self {
        Self { inner: tool }
    }
}

impl From<PyToolDefinition> for ToolDefinition {
    fn from(py_tool: PyToolDefinition) -> Self {
        py_tool.inner
    }
}

/// Builder for creating tool definitions with a fluent API.
///
/// Example:
/// ```python
/// tool = (ToolBuilder("get_weather")
///     .description("Get current weather")
///     .string_param("city", "City name", required=True)
///     .enum_param("unit", "Temperature unit", ["celsius", "fahrenheit"])
///     .build())
/// ```
#[pyclass(name = "ToolBuilder")]
#[derive(Clone)]
pub struct PyToolBuilder {
    inner: ToolBuilder,
}

#[pymethods]
impl PyToolBuilder {
    /// Create a new tool builder.
    ///
    /// Args:
    ///     name: Tool name
    ///
    /// Returns:
    ///     ToolBuilder: A new builder instance
    #[new]
    fn new(name: String) -> Self {
        Self {
            inner: ToolBuilder::new(name),
        }
    }

    /// Set the tool description.
    ///
    /// Args:
    ///     description: Tool description
    ///
    /// Returns:
    ///     ToolBuilder: New builder with description set
    fn description(&self, description: String) -> Self {
        Self {
            inner: self.inner.clone().description(description),
        }
    }

    /// Add a string parameter.
    ///
    /// Args:
    ///     name: Parameter name
    ///     description: Parameter description
    ///     required: Whether the parameter is required
    ///
    /// Returns:
    ///     ToolBuilder: New builder with parameter added
    #[pyo3(signature = (name, description, required = true))]
    fn string_param(&self, name: String, description: String, required: bool) -> Self {
        Self {
            inner: self.inner.clone().string_param(name, description, required),
        }
    }

    /// Add an integer parameter.
    ///
    /// Args:
    ///     name: Parameter name
    ///     description: Parameter description
    ///     required: Whether the parameter is required
    ///
    /// Returns:
    ///     ToolBuilder: New builder with parameter added
    #[pyo3(signature = (name, description, required = true))]
    fn integer_param(&self, name: String, description: String, required: bool) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .integer_param(name, description, required),
        }
    }

    /// Add a number (float) parameter.
    ///
    /// Args:
    ///     name: Parameter name
    ///     description: Parameter description
    ///     required: Whether the parameter is required
    ///
    /// Returns:
    ///     ToolBuilder: New builder with parameter added
    #[pyo3(signature = (name, description, required = true))]
    fn number_param(&self, name: String, description: String, required: bool) -> Self {
        Self {
            inner: self.inner.clone().number_param(name, description, required),
        }
    }

    /// Add a boolean parameter.
    ///
    /// Args:
    ///     name: Parameter name
    ///     description: Parameter description
    ///     required: Whether the parameter is required
    ///
    /// Returns:
    ///     ToolBuilder: New builder with parameter added
    #[pyo3(signature = (name, description, required = true))]
    fn boolean_param(&self, name: String, description: String, required: bool) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .boolean_param(name, description, required),
        }
    }

    /// Add an array parameter.
    ///
    /// Args:
    ///     name: Parameter name
    ///     description: Parameter description
    ///     item_type: Type of array items ("string", "integer", "number", "boolean")
    ///     required: Whether the parameter is required
    ///
    /// Returns:
    ///     ToolBuilder: New builder with parameter added
    #[pyo3(signature = (name, description, item_type, required = true))]
    fn array_param(
        &self,
        name: String,
        description: String,
        item_type: String,
        required: bool,
    ) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .array_param(name, description, &item_type, required),
        }
    }

    /// Add an enum parameter (string with allowed values).
    ///
    /// Args:
    ///     name: Parameter name
    ///     description: Parameter description
    ///     values: List of allowed values
    ///     required: Whether the parameter is required
    ///
    /// Returns:
    ///     ToolBuilder: New builder with parameter added
    #[pyo3(signature = (name, description, values, required = true))]
    fn enum_param(
        &self,
        name: String,
        description: String,
        values: Vec<String>,
        required: bool,
    ) -> Self {
        let values_ref: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        Self {
            inner: self
                .inner
                .clone()
                .enum_param(name, description, &values_ref, required),
        }
    }

    /// Add a custom parameter with a JSON schema.
    ///
    /// Args:
    ///     name: Parameter name
    ///     schema: JSON schema for the parameter
    ///     required: Whether the parameter is required
    ///
    /// Returns:
    ///     ToolBuilder: New builder with custom parameter added
    #[pyo3(signature = (name, schema, required = true))]
    fn custom_param(
        &self,
        name: String,
        schema: Bound<'_, PyDict>,
        required: bool,
    ) -> PyResult<Self> {
        let schema_value: serde_json::Value = pythonize::depythonize(&schema)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: self
                .inner
                .clone()
                .custom_param(name, schema_value, required),
        })
    }

    /// Build the tool definition.
    ///
    /// Returns:
    ///     ToolDefinition: The completed tool definition
    fn build(&self) -> PyToolDefinition {
        PyToolDefinition {
            inner: self.inner.clone().build(),
        }
    }

    fn __repr__(&self) -> String {
        "ToolBuilder(...)".to_string()
    }
}
