use super::traits::{Tool, ToolResult};
use crate::config::{MultimodalConfig, ReliabilityConfig};
use crate::multimodal::analyze_image_refs_to_visual_analysis;
use crate::security::SecurityPolicy;
use async_trait::async_trait;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub struct VisualAnalyzeTool {
    security: Arc<SecurityPolicy>,
    multimodal: MultimodalConfig,
    reliability: ReliabilityConfig,
    workspace_dir: PathBuf,
}

impl VisualAnalyzeTool {
    pub fn new(
        security: Arc<SecurityPolicy>,
        multimodal: MultimodalConfig,
        reliability: ReliabilityConfig,
        workspace_dir: PathBuf,
    ) -> Self {
        Self {
            security,
            multimodal,
            reliability,
            workspace_dir,
        }
    }
}

#[async_trait]
impl Tool for VisualAnalyzeTool {
    fn name(&self) -> &str {
        "analyze_image"
    }

    fn description(&self) -> &str {
        "Run visual analysis on a stored image and return structured content (text, fields, layout). Use when the user's request requires understanding what is inside an image, document scan, receipt, screenshot, or photo. Pass the saved file path and describe what you want to extract."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or workspace-relative path to the image file."
                },
                "request": {
                    "type": "string",
                    "description": "What to look for or extract from the image (e.g. 'describe what you see', 'extract all text', 'identify items and amounts')."
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'path' parameter"))?;

        let request = args
            .get("request")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if !self.security.is_path_allowed(path_str) {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Path not allowed: {path_str}")),
            });
        }

        if !Path::new(path_str).exists() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Image not found: {path_str}")),
            });
        }

        if !self.multimodal.processor.enabled {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(
                    "Visual analysis is not available: multimodal processor is disabled."
                        .to_string(),
                ),
            });
        }

        match analyze_image_refs_to_visual_analysis(
            &request,
            &[path_str.to_string()],
            &self.multimodal,
            &self.reliability,
            Some(&self.workspace_dir),
        )
        .await
        {
            Ok(analysis) => Ok(ToolResult {
                success: true,
                output: analysis,
                error: None,
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Visual analysis failed: {e}")),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::MultimodalProcessorConfig;
    use crate::config::ReliabilityConfig;
    use crate::security::{AutonomyLevel, SecurityPolicy};

    fn make_tool() -> VisualAnalyzeTool {
        let security = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Full,
            workspace_dir: std::env::temp_dir(),
            workspace_only: false,
            forbidden_paths: vec![],
            ..SecurityPolicy::default()
        });
        let multimodal = MultimodalConfig {
            processor: MultimodalProcessorConfig {
                enabled: false,
                ..Default::default()
            },
        };
        let reliability = ReliabilityConfig::default();
        VisualAnalyzeTool::new(security, multimodal, reliability, std::env::temp_dir())
    }

    #[test]
    fn tool_name() {
        assert_eq!(make_tool().name(), "analyze_image");
    }

    #[test]
    fn tool_schema_requires_path() {
        let schema = make_tool().parameters_schema();
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("path")));
    }

    #[tokio::test]
    async fn returns_error_when_processor_disabled() {
        let tool = make_tool();
        let result = tool
            .execute(json!({"path": "/tmp/fake.jpg", "request": "describe"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("disabled"));
    }

    fn make_tool_enabled() -> VisualAnalyzeTool {
        let security = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Full,
            workspace_dir: std::env::temp_dir(),
            workspace_only: false,
            forbidden_paths: vec![],
            ..SecurityPolicy::default()
        });
        let multimodal = MultimodalConfig {
            processor: MultimodalProcessorConfig {
                enabled: true,
                ..Default::default()
            },
            ..Default::default()
        };
        VisualAnalyzeTool::new(security, multimodal, ReliabilityConfig::default(), std::env::temp_dir())
    }

    #[tokio::test]
    async fn returns_error_for_missing_file() {
        let tool = make_tool_enabled();
        let result = tool
            .execute(json!({"path": "/tmp/does_not_exist_xyz.jpg"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("not found"));
    }
}
