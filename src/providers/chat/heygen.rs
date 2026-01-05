//! HeyGen provider for AI avatar and video generation.
//!
//! HeyGen provides AI-powered avatar videos and talking head generation.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// HeyGen provider
pub struct HeyGenProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl HeyGenProvider {
    /// Create a new HeyGen provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.heygen.com/v2".to_string(),
        }
    }

    /// Create with custom base URL
    pub fn with_base_url(api_key: &str, base_url: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: base_url.to_string(),
        }
    }

    /// Create from environment variable
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("HEYGEN_API_KEY")
            .map_err(|_| Error::Configuration("HEYGEN_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available features
    pub async fn list_features(&self) -> Result<Vec<String>> {
        Ok(vec![
            "avatar-video".to_string(),
            "streaming-avatar".to_string(),
            "video-translate".to_string(),
            "photo-avatar".to_string(),
        ])
    }

    /// Get feature details
    pub fn get_feature_info(feature: &str) -> Option<HeyGenFeatureInfo> {
        match feature {
            "avatar-video" => Some(HeyGenFeatureInfo {
                name: feature.to_string(),
                description: "Generate videos with AI avatars".to_string(),
                inputs: vec![
                    "script".to_string(),
                    "avatar_id".to_string(),
                    "voice_id".to_string(),
                ],
                max_duration_seconds: 300,
                languages: 40,
            }),
            "streaming-avatar" => Some(HeyGenFeatureInfo {
                name: feature.to_string(),
                description: "Real-time interactive AI avatar".to_string(),
                inputs: vec!["avatar_id".to_string(), "voice_id".to_string()],
                max_duration_seconds: 0, // Real-time
                languages: 40,
            }),
            "video-translate" => Some(HeyGenFeatureInfo {
                name: feature.to_string(),
                description: "Translate videos with lip-sync".to_string(),
                inputs: vec!["video".to_string(), "target_language".to_string()],
                max_duration_seconds: 600,
                languages: 40,
            }),
            "photo-avatar" => Some(HeyGenFeatureInfo {
                name: feature.to_string(),
                description: "Create avatar from photo".to_string(),
                inputs: vec!["photo".to_string()],
                max_duration_seconds: 0,
                languages: 40,
            }),
            _ => None,
        }
    }
}

/// HeyGen feature information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeyGenFeatureInfo {
    pub name: String,
    pub description: String,
    pub inputs: Vec<String>,
    pub max_duration_seconds: u32,
    pub languages: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = HeyGenProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.heygen.com/v2");
    }

    #[tokio::test]
    async fn test_list_features() {
        let provider = HeyGenProvider::new("test-key");
        let features = provider.list_features().await.unwrap();
        assert!(features.len() >= 3);
    }

    #[test]
    fn test_feature_info() {
        let info = HeyGenProvider::get_feature_info("avatar-video").unwrap();
        assert_eq!(info.languages, 40);
    }
}
