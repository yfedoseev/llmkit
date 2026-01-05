//! Petals provider for distributed LLM inference.
//!
//! Petals provides distributed inference across multiple machines.

use crate::error::Result;
use serde::{Deserialize, Serialize};

/// Petals provider
pub struct PetalsProvider {
    #[allow(dead_code)]
    base_url: String,
}

impl PetalsProvider {
    /// Create a new Petals provider
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
        }
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "petals-team/StableBeluga2".to_string(),
            "bigscience/bloomz".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<PetalsModelInfo> {
        match model {
            m if m.contains("StableBeluga2") => Some(PetalsModelInfo {
                name: model.to_string(),
                context_window: 4096,
                distributed: true,
            }),
            m if m.contains("bloomz") => Some(PetalsModelInfo {
                name: model.to_string(),
                context_window: 2048,
                distributed: true,
            }),
            _ => None,
        }
    }
}

/// Petals model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PetalsModelInfo {
    pub name: String,
    pub context_window: u32,
    pub distributed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = PetalsProvider::new("http://localhost:8080");
        assert_eq!(provider.base_url, "http://localhost:8080");
    }
}
