use anyhow::Context;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HtmlOutputFormat {
    Markdown,
    Text,
    Html,
    RawHtml,
}

impl HtmlOutputFormat {
    pub fn parse(raw: Option<&str>) -> anyhow::Result<Self> {
        match raw
            .unwrap_or("markdown")
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "" | "markdown" | "md" => Ok(Self::Markdown),
            "text" | "plain" | "plain_text" => Ok(Self::Text),
            "html" => Ok(Self::Html),
            "raw_html" | "raw" => Ok(Self::RawHtml),
            other => {
                anyhow::bail!(
                    "Unsupported HTML output format '{other}'. Use 'markdown', 'text', 'html', or 'raw_html'."
                )
            }
        }
    }
}

pub fn extract_html_for_llm(
    html: &str,
    source_url: &str,
    format: HtmlOutputFormat,
) -> anyhow::Result<String> {
    match format {
        HtmlOutputFormat::RawHtml => Ok(html.to_string()),
        HtmlOutputFormat::Html => {
            let extracted = extract_main_content(html, source_url, true);
            html_from_extraction(html, extracted.as_ref())
        }
        HtmlOutputFormat::Markdown => {
            let extracted = extract_main_content(html, source_url, false);
            markdown_from_extraction(html, extracted.as_ref())
        }
        HtmlOutputFormat::Text => {
            let extracted = extract_main_content(html, source_url, false);
            text_from_extraction(html, extracted.as_ref())
        }
    }
}

fn extract_main_content(
    html: &str,
    source_url: &str,
    include_images: bool,
) -> Option<rs_trafilatura::ExtractResult> {
    let options = rs_trafilatura::Options {
        include_tables: true,
        include_links: true,
        include_images,
        include_comments: false,
        include_title_in_content: true,
        min_extracted_size: 1,
        min_extracted_len: 1,
        min_output_size: 1,
        use_fallback_extraction: true,
        url: Some(source_url.to_string()),
        ..rs_trafilatura::Options::default()
    };

    match rs_trafilatura::extract_with_options(html, &options) {
        Ok(result) => Some(result),
        Err(err) => {
            tracing::debug!("rs-trafilatura extraction failed, falling back to full HTML: {err}");
            None
        }
    }
}

fn html_from_extraction(
    original_html: &str,
    extracted: Option<&rs_trafilatura::ExtractResult>,
) -> anyhow::Result<String> {
    if let Some(extracted_html) = extracted.and_then(|result| non_empty(result.content_html.as_deref())) {
        return Ok(extracted_html.to_string());
    }

    Ok(original_html.to_string())
}

fn markdown_from_extraction(
    original_html: &str,
    extracted: Option<&rs_trafilatura::ExtractResult>,
) -> anyhow::Result<String> {
    let extracted_html = extracted
        .and_then(|result| non_empty(result.content_html.as_deref()))
        .filter(|html| !html.is_empty());
    let primary_html = extracted_html.unwrap_or(original_html);

    let markdown = convert_html_to_markdown(primary_html)
        .or_else(|primary_err| {
            if extracted_html.is_none() {
                Err(primary_err)
            } else {
                tracing::debug!(
                    "HTML to Markdown conversion failed for extracted content, falling back to full HTML: {primary_err}"
                );
                convert_html_to_markdown(original_html)
            }
        })
        .context("HTML to Markdown conversion failed")?;

    if let Some(markdown) = non_empty(Some(markdown.as_str())) {
        return Ok(markdown.to_string());
    }

    if let Some(text) = extracted.and_then(|result| non_empty(Some(result.content_text.as_str()))) {
        return Ok(text.to_string());
    }

    Ok(markdown)
}

fn text_from_extraction(
    original_html: &str,
    extracted: Option<&rs_trafilatura::ExtractResult>,
) -> anyhow::Result<String> {
    if let Some(text) = extracted.and_then(|result| non_empty(Some(result.content_text.as_str()))) {
        return Ok(text.to_string());
    }

    convert_html_to_markdown(original_html).context("HTML to text fallback conversion failed")
}

fn convert_html_to_markdown(html: &str) -> anyhow::Result<String> {
    let result = html_to_markdown_rs::convert(html, None)?;
    Ok(result.content.unwrap_or_default())
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_format_aliases() {
        assert_eq!(
            HtmlOutputFormat::parse(None).unwrap(),
            HtmlOutputFormat::Markdown
        );
        assert_eq!(
            HtmlOutputFormat::parse(Some("md")).unwrap(),
            HtmlOutputFormat::Markdown
        );
        assert_eq!(
            HtmlOutputFormat::parse(Some("plain_text")).unwrap(),
            HtmlOutputFormat::Text
        );
        assert_eq!(
            HtmlOutputFormat::parse(Some("html")).unwrap(),
            HtmlOutputFormat::Html
        );
        assert_eq!(
            HtmlOutputFormat::parse(Some("raw")).unwrap(),
            HtmlOutputFormat::RawHtml
        );
        assert!(HtmlOutputFormat::parse(Some("xml")).is_err());
    }

    #[test]
    fn converts_html_to_markdown_without_custom_renderer() {
        let html = "<html><body><main><h1>Title</h1><p>Hello <strong>world</strong></p></main></body></html>";
        let markdown =
            extract_html_for_llm(html, "https://example.com/page", HtmlOutputFormat::Markdown)
                .unwrap();

        assert!(markdown.contains("Title"));
        assert!(markdown.contains("Hello"));
        assert!(markdown.contains("world"));
        assert!(!markdown.contains("<h1>"));
        assert!(!markdown.contains("<p>"));
    }

    #[test]
    fn text_format_uses_extracted_visible_content() {
        let html =
            "<html><body><article><h1>Title</h1><p>Hello <b>world</b></p></article></body></html>";
        let text =
            extract_html_for_llm(html, "https://example.com/page", HtmlOutputFormat::Text).unwrap();

        assert!(text.contains("Title"));
        assert!(text.contains("Hello"));
        assert!(text.contains("world"));
        assert!(!text.contains("<article>"));
    }

    #[test]
    fn html_format_preserves_image_tags() {
        let html = "<html><body><article><h1>Title</h1><img src=\"https://example.com/hero.jpg\" alt=\"hero\" /><p>Hello world</p></article></body></html>";
        let extracted =
            extract_html_for_llm(html, "https://example.com/page", HtmlOutputFormat::Html)
                .unwrap();

        assert!(extracted.contains("<img"));
        assert!(extracted.contains("hero.jpg"));
        assert!(extracted.contains("Hello world"));
    }

    #[test]
    fn raw_html_format_returns_original_document() {
        let html = "<html><head><title>Title</title></head><body><article><img src=\"hero.jpg\" /></article></body></html>";
        let extracted =
            extract_html_for_llm(html, "https://example.com/page", HtmlOutputFormat::RawHtml)
                .unwrap();

        assert_eq!(extracted, html);
    }

    #[test]
    fn markdown_falls_back_for_short_non_article_pages() {
        let html = "<html><body><div>Tiny page</div></body></html>";
        let markdown =
            extract_html_for_llm(html, "https://example.com/page", HtmlOutputFormat::Markdown)
                .unwrap();

        assert!(markdown.contains("Tiny page"));
    }
}
