//! Prompt template infrastructure for variable substitution.
//!
//! This module provides a flexible template system for building prompts
//! with variable placeholders that can be filled at runtime.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::{PromptTemplate, TemplateRegistry};
//! use std::collections::HashMap;
//!
//! // Create a template
//! let template = PromptTemplate::new(
//!     "You are a {{role}}. Please help the user with: {{task}}"
//! );
//!
//! // Check required variables
//! assert!(template.variables().contains("role"));
//! assert!(template.variables().contains("task"));
//!
//! // Render with values
//! let mut values = HashMap::new();
//! values.insert("role".to_string(), "helpful assistant".to_string());
//! values.insert("task".to_string(), "writing code".to_string());
//!
//! let prompt = template.render(&values)?;
//! // "You are a helpful assistant. Please help the user with: writing code"
//! ```
//!
//! # Template Registry
//!
//! For managing multiple templates:
//!
//! ```ignore
//! let mut registry = TemplateRegistry::new();
//!
//! registry.register("greeting", PromptTemplate::new("Hello, {{name}}!"));
//! registry.register("task", PromptTemplate::new("Please {{action}} the {{target}}."));
//!
//! let greeting = registry.render("greeting", &HashMap::from([
//!     ("name".to_string(), "Alice".to_string()),
//! ]))?;
//! ```

use std::collections::{HashMap, HashSet};

use regex::Regex;

use crate::error::{Error, Result};
use crate::types::{CompletionRequest, Message};

/// A prompt template with variable placeholders.
///
/// Variables are denoted with double curly braces: `{{variable_name}}`
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    /// The raw template string.
    template: String,
    /// Extracted variable names.
    variables: HashSet<String>,
}

impl PromptTemplate {
    /// Create a new template, extracting variable placeholders.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let template = PromptTemplate::new("Hello, {{name}}!");
    /// assert!(template.variables().contains("name"));
    /// ```
    pub fn new(template: impl Into<String>) -> Self {
        let template = template.into();
        let variables = Self::extract_variables(&template);
        Self {
            template,
            variables,
        }
    }

    /// Extract variable names from a template string.
    fn extract_variables(template: &str) -> HashSet<String> {
        // Match {{variable_name}} patterns
        let re = Regex::new(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}").unwrap();
        re.captures_iter(template)
            .map(|cap| cap[1].to_string())
            .collect()
    }

    /// Get the set of required variables.
    pub fn variables(&self) -> &HashSet<String> {
        &self.variables
    }

    /// Check if this template has any variables.
    pub fn has_variables(&self) -> bool {
        !self.variables.is_empty()
    }

    /// Get the raw template string.
    pub fn raw(&self) -> &str {
        &self.template
    }

    /// Render the template with the given values.
    ///
    /// # Errors
    ///
    /// Returns an error if any required variable is missing from the values map.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let template = PromptTemplate::new("Hello, {{name}}!");
    /// let mut values = HashMap::new();
    /// values.insert("name".to_string(), "World".to_string());
    /// assert_eq!(template.render(&values)?, "Hello, World!");
    /// ```
    pub fn render(&self, values: &HashMap<String, String>) -> Result<String> {
        // Check for missing variables
        let missing: Vec<_> = self
            .variables
            .iter()
            .filter(|v| !values.contains_key(*v))
            .collect();

        if !missing.is_empty() {
            return Err(Error::other(format!(
                "Missing template variables: {}",
                missing
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }

        // Replace all variables
        let mut result = self.template.clone();
        for (name, value) in values {
            let placeholder = format!("{{{{{}}}}}", name);
            result = result.replace(&placeholder, value);
        }

        Ok(result)
    }

    /// Render the template, using defaults for missing variables.
    ///
    /// Variables not in `values` will be replaced with empty string or
    /// their value from `defaults` if provided.
    pub fn render_with_defaults(
        &self,
        values: &HashMap<String, String>,
        defaults: &HashMap<String, String>,
    ) -> String {
        let mut result = self.template.clone();
        for var in &self.variables {
            let placeholder = format!("{{{{{}}}}}", var);
            let value = values
                .get(var)
                .or_else(|| defaults.get(var))
                .map(|s| s.as_str())
                .unwrap_or("");
            result = result.replace(&placeholder, value);
        }
        result
    }

    /// Render keeping unmatched variables as-is.
    ///
    /// Useful for partial template expansion.
    pub fn render_partial(&self, values: &HashMap<String, String>) -> String {
        let mut result = self.template.clone();
        for (name, value) in values {
            let placeholder = format!("{{{{{}}}}}", name);
            result = result.replace(&placeholder, value);
        }
        result
    }

    /// Validate that all required variables are present.
    pub fn validate(&self, values: &HashMap<String, String>) -> Result<()> {
        let missing: Vec<_> = self
            .variables
            .iter()
            .filter(|v| !values.contains_key(*v))
            .collect();

        if missing.is_empty() {
            Ok(())
        } else {
            Err(Error::other(format!(
                "Missing template variables: {}",
                missing
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )))
        }
    }
}

/// A registry for managing named templates.
#[derive(Debug, Clone, Default)]
pub struct TemplateRegistry {
    templates: HashMap<String, PromptTemplate>,
}

impl TemplateRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a template with a name.
    pub fn register(&mut self, name: impl Into<String>, template: PromptTemplate) {
        self.templates.insert(name.into(), template);
    }

    /// Register a template from a string.
    pub fn register_str(&mut self, name: impl Into<String>, template: impl Into<String>) {
        self.templates
            .insert(name.into(), PromptTemplate::new(template));
    }

    /// Get a template by name.
    pub fn get(&self, name: &str) -> Option<&PromptTemplate> {
        self.templates.get(name)
    }

    /// Check if a template exists.
    pub fn contains(&self, name: &str) -> bool {
        self.templates.contains_key(name)
    }

    /// Remove a template by name.
    pub fn remove(&mut self, name: &str) -> Option<PromptTemplate> {
        self.templates.remove(name)
    }

    /// Get the number of registered templates.
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }

    /// List all template names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.templates.keys().map(|s| s.as_str())
    }

    /// Render a template by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the template doesn't exist or if required variables are missing.
    pub fn render(&self, name: &str, values: &HashMap<String, String>) -> Result<String> {
        self.templates
            .get(name)
            .ok_or_else(|| Error::other(format!("Template not found: {}", name)))?
            .render(values)
    }

    /// Get variables required by a template.
    pub fn variables(&self, name: &str) -> Option<&HashSet<String>> {
        self.templates.get(name).map(|t| t.variables())
    }
}

/// Builder for creating CompletionRequest from templates.
#[derive(Debug, Clone)]
pub struct TemplatedRequestBuilder {
    model: String,
    system_template: Option<PromptTemplate>,
    user_template: Option<PromptTemplate>,
    values: HashMap<String, String>,
}

impl TemplatedRequestBuilder {
    /// Create a new builder with the specified model.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            system_template: None,
            user_template: None,
            values: HashMap::new(),
        }
    }

    /// Set the system prompt template.
    pub fn system_template(mut self, template: PromptTemplate) -> Self {
        self.system_template = Some(template);
        self
    }

    /// Set the system prompt template from a string.
    pub fn system_template_str(mut self, template: impl Into<String>) -> Self {
        self.system_template = Some(PromptTemplate::new(template));
        self
    }

    /// Set the user message template.
    pub fn user_template(mut self, template: PromptTemplate) -> Self {
        self.user_template = Some(template);
        self
    }

    /// Set the user message template from a string.
    pub fn user_template_str(mut self, template: impl Into<String>) -> Self {
        self.user_template = Some(PromptTemplate::new(template));
        self
    }

    /// Set a template variable value.
    pub fn var(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.values.insert(name.into(), value.into());
        self
    }

    /// Set multiple template variable values.
    pub fn vars(mut self, values: HashMap<String, String>) -> Self {
        self.values.extend(values);
        self
    }

    /// Get required variables from all templates.
    pub fn required_variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        if let Some(ref t) = self.system_template {
            vars.extend(t.variables().clone());
        }
        if let Some(ref t) = self.user_template {
            vars.extend(t.variables().clone());
        }
        vars
    }

    /// Check if all required variables are set.
    pub fn is_complete(&self) -> bool {
        self.required_variables()
            .iter()
            .all(|v| self.values.contains_key(v))
    }

    /// Build the completion request.
    ///
    /// # Errors
    ///
    /// Returns an error if required variables are missing or if no user template is set.
    pub fn build(self) -> Result<CompletionRequest> {
        let user_template = self
            .user_template
            .ok_or_else(|| Error::other("User template is required"))?;

        let user_content = user_template.render(&self.values)?;
        let mut request = CompletionRequest::new(&self.model, vec![Message::user(user_content)]);

        if let Some(system_template) = self.system_template {
            let system_content = system_template.render(&self.values)?;
            request = request.with_system(system_content);
        }

        Ok(request)
    }
}

/// Convenience functions for common template patterns.
pub mod patterns {
    use super::PromptTemplate;

    /// Create a simple Q&A template.
    pub fn qa_template() -> PromptTemplate {
        PromptTemplate::new("Question: {{question}}\n\nPlease provide a detailed answer.")
    }

    /// Create a summarization template.
    pub fn summarization_template() -> PromptTemplate {
        PromptTemplate::new("Please summarize the following text:\n\n{{text}}\n\nSummary:")
    }

    /// Create a translation template.
    pub fn translation_template() -> PromptTemplate {
        PromptTemplate::new(
            "Translate the following text from {{source_language}} to {{target_language}}:\n\n{{text}}\n\nTranslation:",
        )
    }

    /// Create a code explanation template.
    pub fn code_explanation_template() -> PromptTemplate {
        PromptTemplate::new(
            "Explain what the following {{language}} code does:\n\n```{{language}}\n{{code}}\n```\n\nExplanation:",
        )
    }

    /// Create a code generation template.
    pub fn code_generation_template() -> PromptTemplate {
        PromptTemplate::new(
            "Write {{language}} code that {{task}}.\n\nRequirements:\n{{requirements}}\n\nCode:",
        )
    }

    /// Create a classification template.
    pub fn classification_template() -> PromptTemplate {
        PromptTemplate::new(
            "Classify the following text into one of these categories: {{categories}}\n\nText: {{text}}\n\nCategory:",
        )
    }

    /// Create a chat system prompt template.
    pub fn chat_system_template() -> PromptTemplate {
        PromptTemplate::new("You are {{name}}, a {{role}}. {{personality}}")
    }

    /// Create a RAG (Retrieval Augmented Generation) template.
    pub fn rag_template() -> PromptTemplate {
        PromptTemplate::new(
            "Use the following context to answer the question. If the answer is not in the context, say so.\n\nContext:\n{{context}}\n\nQuestion: {{question}}\n\nAnswer:",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_extraction() {
        let template = PromptTemplate::new("Hello, {{name}}! You are a {{role}}.");
        assert_eq!(template.variables().len(), 2);
        assert!(template.variables().contains("name"));
        assert!(template.variables().contains("role"));
    }

    #[test]
    fn test_template_no_variables() {
        let template = PromptTemplate::new("Hello, World!");
        assert!(template.variables().is_empty());
        assert!(!template.has_variables());
    }

    #[test]
    fn test_template_render() {
        let template = PromptTemplate::new("Hello, {{name}}!");
        let mut values = HashMap::new();
        values.insert("name".to_string(), "Alice".to_string());

        let result = template.render(&values).unwrap();
        assert_eq!(result, "Hello, Alice!");
    }

    #[test]
    fn test_template_render_multiple() {
        let template = PromptTemplate::new("{{greeting}}, {{name}}! Welcome to {{place}}.");
        let mut values = HashMap::new();
        values.insert("greeting".to_string(), "Hello".to_string());
        values.insert("name".to_string(), "Bob".to_string());
        values.insert("place".to_string(), "Rust".to_string());

        let result = template.render(&values).unwrap();
        assert_eq!(result, "Hello, Bob! Welcome to Rust.");
    }

    #[test]
    fn test_template_render_missing_variable() {
        let template = PromptTemplate::new("Hello, {{name}}!");
        let values = HashMap::new();

        let result = template.render(&values);
        assert!(result.is_err());
    }

    #[test]
    fn test_template_render_partial() {
        let template = PromptTemplate::new("{{greeting}}, {{name}}!");
        let mut values = HashMap::new();
        values.insert("greeting".to_string(), "Hi".to_string());

        let result = template.render_partial(&values);
        assert_eq!(result, "Hi, {{name}}!");
    }

    #[test]
    fn test_template_render_with_defaults() {
        let template = PromptTemplate::new("{{greeting}}, {{name}}!");
        let values = HashMap::new();
        let mut defaults = HashMap::new();
        defaults.insert("greeting".to_string(), "Hello".to_string());
        defaults.insert("name".to_string(), "World".to_string());

        let result = template.render_with_defaults(&values, &defaults);
        assert_eq!(result, "Hello, World!");
    }

    #[test]
    fn test_template_validate() {
        let template = PromptTemplate::new("{{a}} {{b}}");
        let mut values = HashMap::new();
        values.insert("a".to_string(), "1".to_string());

        assert!(template.validate(&values).is_err());

        values.insert("b".to_string(), "2".to_string());
        assert!(template.validate(&values).is_ok());
    }

    #[test]
    fn test_registry() {
        let mut registry = TemplateRegistry::new();
        registry.register_str("greeting", "Hello, {{name}}!");
        registry.register_str("farewell", "Goodbye, {{name}}!");

        assert!(registry.contains("greeting"));
        assert!(!registry.contains("other"));
        assert_eq!(registry.len(), 2);

        let mut values = HashMap::new();
        values.insert("name".to_string(), "Alice".to_string());

        assert_eq!(
            registry.render("greeting", &values).unwrap(),
            "Hello, Alice!"
        );
        assert_eq!(
            registry.render("farewell", &values).unwrap(),
            "Goodbye, Alice!"
        );
    }

    #[test]
    fn test_registry_not_found() {
        let registry = TemplateRegistry::new();
        let values = HashMap::new();

        assert!(registry.render("nonexistent", &values).is_err());
    }

    #[test]
    fn test_templated_request_builder() {
        let request = TemplatedRequestBuilder::new("gpt-4")
            .system_template_str("You are a {{role}}.")
            .user_template_str("Please help me with: {{task}}")
            .var("role", "helpful assistant")
            .var("task", "writing tests")
            .build()
            .unwrap();

        assert_eq!(request.model, "gpt-4");
        assert!(request.system.is_some());
        assert_eq!(request.system.unwrap(), "You are a helpful assistant.");
        assert_eq!(request.messages.len(), 1);
    }

    #[test]
    fn test_templated_request_missing_user() {
        let result = TemplatedRequestBuilder::new("gpt-4")
            .system_template_str("System prompt")
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_patterns() {
        let qa = patterns::qa_template();
        assert!(qa.variables().contains("question"));

        let translation = patterns::translation_template();
        assert!(translation.variables().contains("source_language"));
        assert!(translation.variables().contains("target_language"));
        assert!(translation.variables().contains("text"));

        let rag = patterns::rag_template();
        assert!(rag.variables().contains("context"));
        assert!(rag.variables().contains("question"));
    }

    #[test]
    fn test_complex_variable_names() {
        let template = PromptTemplate::new("{{var_1}} {{var_2}} {{_private}}");
        assert_eq!(template.variables().len(), 3);
        assert!(template.variables().contains("var_1"));
        assert!(template.variables().contains("var_2"));
        assert!(template.variables().contains("_private"));
    }

    #[test]
    fn test_repeated_variables() {
        let template = PromptTemplate::new("{{name}} said hello. {{name}} is happy.");
        // Should only have one unique variable
        assert_eq!(template.variables().len(), 1);

        let mut values = HashMap::new();
        values.insert("name".to_string(), "Alice".to_string());

        let result = template.render(&values).unwrap();
        assert_eq!(result, "Alice said hello. Alice is happy.");
    }
}
