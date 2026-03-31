use std::collections::HashSet;

pub fn build_product_delivery_guidance(tool_names: &[&str]) -> Option<String> {
    let names: HashSet<&str> = tool_names.iter().copied().collect();

    let has_web_fetch = names.contains("web_fetch");
    let has_text_browser = names.contains("text_browser");
    let has_browser_open = names.contains("browser_open") || names.contains("browser_delegate");
    let has_http = names.contains("http_request");
    let has_read = names.contains("file_read");
    let has_write = names.contains("file_write");
    let has_edit = names.contains("file_edit");
    let has_glob = names.contains("glob_search");
    let has_search = names.contains("content_search");
    let has_shell = names.contains("shell");

    let has_site_analysis = has_web_fetch || has_text_browser || has_browser_open || has_http;
    let has_product_artifacts = has_read || has_write || has_edit || has_glob || has_search;

    if !(has_site_analysis || has_product_artifacts || has_shell) {
        return None;
    }

    let mut out = String::from(
        "## Product & Site Delivery\n\n\
         When the user asks you to evaluate an existing site, design a new product, or iterate \
         on a concrete specification, behave like a product-delivery agent: inspect references, \
         maintain a living spec, ship a real version, and record what changed.\n\n",
    );

    if has_site_analysis {
        out.push_str(
            "- If the user gives you a reference URL, inspect it before proposing or building the new version.\n",
        );
        if has_web_fetch {
            out.push_str(
                "- Use `web_fetch` on the homepage and a few important linked pages to extract information architecture, content hierarchy, and reusable cues.\n",
            );
        }
        if has_text_browser || has_browser_open {
            out.push_str(
                "- When layout or visual flow matters, inspect the live site in a browser-oriented tool instead of relying on memory.\n",
            );
        }
        out.push_str(
            "- If live inspection fails, switch explicitly into `inference mode`: say what you could not inspect, separate evidence from assumptions, and continue with clearly labeled assumptions instead of hallucinating certainty.\n",
        );
    }

    if has_product_artifacts {
        out.push_str(
            "- Capture reference-site findings in `product/analysis/` so later iterations do not lose the original reasoning.\n\
             - Treat `product/specs/current.md` as the living source of truth for the current product: goal, audience, constraints, references, accepted decisions, and open questions.\n\
             - Store handoffs for other agents in `product/handoffs/` and reusable build/style notes in `product/approaches/`.\n\
             - Record each concrete delivery or iteration in `product/revisions/` using monotonic version notes such as `v1.md`, `v2.md`, and explain what changed.\n\
             - Do not describe a site analysis artifact as if it were the shipped product; analysis files and delivered versions are different things.\n",
        );
        out.push_str(
            "- When the user clearly changes topic, company, or reference, treat that as a likely project switch instead of blindly iterating the previous direction.\n\
             - Use `services/` for background workers, sync jobs, webhook handlers, cron tasks, or small APIs that belong to the current project but are not the public site.\n",
        );
        out.push_str(
            "- Product decomposition outputs should be explicit and structured. When relevant, produce sections for: screens, states, entities, flows, rules, risks, technical assumptions, and open questions.\n",
        );
        out.push_str(
            "- Handoff outputs for another agent should be explicit and structured. When relevant, produce sections for: spec, tasklist, component map, visual criteria, engineering decisions, and risks.\n",
        );
    }

    if has_write || has_edit {
        out.push_str(
            "- If the user changes direction entirely (new product, replace, redesign from scratch), rewrite the living spec intentionally and note that the next revision replaces the previous direction.\n\
             - If the user asks for a refinement or a new version, update the spec first, then implement the requested delta instead of ignoring prior context.\n\
             - After an analysis-only turn, close proactively: summarize the evidence, say the analysis/spec are ready, and explicitly offer to build the first version next.\n\
             - If the user follows the analysis with `implement it`, `build it`, `advance`, `dale`, or similar, treat that as authorization to ship the first version now instead of merely writing another note.\n",
        );
        out.push_str(
            "- Choose and record an explicit build approach instead of defaulting to one generic output. Common approaches include `corporate_marketing`, `editorial_brand`, `minimal_landing`, `dashboard`, `storefront`, and `raw_custom`.\n\
             - Also choose and record the target you intend to ship, such as `static_html`, `react_app`, `next_app`, or `portable_handoff`.\n",
        );
    }

    if has_shell {
        out.push_str(
            "- Verify each shipped version with at least one concrete check when possible: build, serve, targeted smoke test, or diff inspection.\n",
        );
        out.push_str(
            "- If the user asks for a service, worker, sync process, webhook handler, or cron job, you may install standard dependencies, scaffold files under `services/`, and run the service locally when the request requires it.\n",
        );
    }

    out.push_str(
        "- Prefer a working v1 over endless clarification loops: analyze the reference, write down the spec, ship a first version, then iterate with explicit deltas.\n\
         - Do not claim a new version exists unless you updated files or verified that the requested version already matches the current state.\n",
    );

    Some(out)
}

#[cfg(test)]
mod tests {
    use super::build_product_delivery_guidance;

    #[test]
    fn product_guidance_requires_relevant_tools() {
        assert!(build_product_delivery_guidance(&["memory_recall"]).is_none());
    }

    #[test]
    fn product_guidance_mentions_reference_sites_and_specs() {
        let rendered = build_product_delivery_guidance(&[
            "web_fetch",
            "browser_open",
            "file_read",
            "file_write",
            "file_edit",
            "glob_search",
            "content_search",
            "shell",
        ])
        .expect("product workflow should be enabled");

        assert!(rendered.contains("## Product & Site Delivery"));
        assert!(rendered.contains("reference URL"));
        assert!(rendered.contains("product/specs/current.md"));
        assert!(rendered.contains("product/revisions/"));
        assert!(rendered.contains("product/handoffs/"));
        assert!(rendered.contains("working v1"));
        assert!(rendered.contains("analysis/spec are ready"));
        assert!(rendered.contains("ship the first version now"));
        assert!(rendered.contains("corporate_marketing"));
        assert!(rendered.contains("inference mode"));
    }
}
