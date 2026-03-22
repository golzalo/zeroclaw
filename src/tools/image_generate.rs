use super::linkedin_client::ImageGenerator;
use super::traits::{Tool, ToolResult};
use crate::config::LinkedInImageConfig;
use crate::security::SecurityPolicy;
use async_trait::async_trait;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;

/// Generic image generation tool that reuses the multi-provider pipeline
/// (Stability, Imagen, DALL·E, Flux) defined for LinkedIn posts.
pub struct ImageGenerateTool {
    security: Arc<SecurityPolicy>,
    workspace_dir: PathBuf,
    image_config: LinkedInImageConfig,
}

impl ImageGenerateTool {
    pub fn new(
        security: Arc<SecurityPolicy>,
        workspace_dir: PathBuf,
        image_config: LinkedInImageConfig,
    ) -> Self {
        Self {
            security,
            workspace_dir,
            image_config,
        }
    }

    fn is_enabled(&self) -> bool {
        self.image_config.enabled
    }
}

#[async_trait]
impl Tool for ImageGenerateTool {
    fn name(&self) -> &str {
        "image.generate"
    }

    fn description(&self) -> &str {
        "Generate an image via the configured Stability/Imagen/DALL·E/Flux providers (see [linkedin.image]). Returns a `[IMAGE:<path>]` marker pointing to the saved file."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text prompt describing the desired image."
                }
            },
            "required": ["prompt"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        if !self.security.can_act() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("Action blocked: autonomy is read-only".into()),
            });
        }

        if !self.security.record_action() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("Action blocked: rate limit exceeded".into()),
            });
        }

        let prompt = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty());

        let Some(prompt) = prompt else {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("Missing required 'prompt' string".into()),
            });
        };

        if !self.is_enabled() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("Image generation is disabled. Set [linkedin.image] enabled=true in config.toml.".into()),
            });
        }

        let mut config = self.image_config.clone();
        // For this generic tool we prefer to surface provider failures instead of
        // emitting SVG fallback cards that channels may not support.
        config.fallback_card = false;

        let generator = ImageGenerator::new(config, self.workspace_dir.clone());
        match generator.generate(prompt).await {
            Ok(path) => Ok(ToolResult {
                success: true,
                output: format!("[IMAGE:{}]", path.display()),
                error: None,
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Image generation failed: {e}")),
            }),
        }
    }
}
