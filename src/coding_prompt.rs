use std::collections::HashSet;

pub fn build_coding_work_guidance(tool_names: &[&str]) -> Option<String> {
    let names: HashSet<&str> = tool_names.iter().copied().collect();

    let has_shell = names.contains("shell");
    let has_read = names.contains("file_read");
    let has_write = names.contains("file_write");
    let has_edit = names.contains("file_edit");
    let has_glob = names.contains("glob_search");
    let has_search = names.contains("content_search");
    let has_git = names.contains("git_operations");
    let has_delegate = names.contains("delegate") || names.contains("swarm");

    if !(has_shell
        || has_read
        || has_write
        || has_edit
        || has_glob
        || has_search
        || has_git
        || has_delegate)
    {
        return None;
    }

    let mut out = String::from(
        "## Coding Work\n\n\
         When the user asks you to build, fix, refactor, review, or explain code in a repository, \
         behave like an engineering agent: inspect the codebase, make changes, verify them, and \
         report concrete results.\n\n\
         - Prefer action over questionnaires. If you can build a sensible v1 from the current request, start working.\n",
    );

    if has_glob || has_search {
        out.push_str(
            "- Use targeted search first to find the right files quickly before reading or editing large areas.\n",
        );
    }
    if has_read {
        out.push_str(
            "- Read the current implementation before changing behavior so your edits match the existing code.\n",
        );
    }
    if has_edit {
        out.push_str(
            "- Use `file_edit` for precise, single-location patches when the existing text can be matched exactly once.\n",
        );
    }
    if has_write {
        out.push_str(
            "- Use `file_write` for new files and deliberate full rewrites; avoid whole-file rewrites when a focused edit is safer.\n",
        );
    }
    if has_shell {
        out.push_str(
            "- Use `shell` for targeted repo work: tests, builds, linters, `git diff`, and local diagnostics.\n",
        );
    }
    if has_git {
        out.push_str(
            "- Use `git_operations` for repository-aware status, diff, and history checks when that is clearer or safer than raw shell commands.\n",
        );
    }
    if has_delegate {
        out.push_str(
            "- If the task is large, exploratory, or likely to need multiple passes, use `delegate` or `swarm`; do not delegate tiny one-line edits.\n",
        );
    }

    out.push_str(
        "- For quick/basic apps or prototypes, ship a sensible first version and iterate instead of asking for extensive upfront specs.\n\
         - If the workspace already contains project-specific build, serve, or publish flows, use them instead of assuming the user must do those steps manually.\n\
         - For longer coding tasks, send one short start update, then only real milestone or blocker updates.\n\
         - Never claim code is done unless you changed files or verified that the requested behavior already exists.\n\
         - Before closing, perform at least one concrete verification when possible: targeted test, build, lint, or diff inspection.\n\
         - Completion summaries must include changed files or areas, verification status, and remaining caveats.\n",
    );

    Some(out)
}

#[cfg(test)]
mod tests {
    use super::build_coding_work_guidance;

    #[test]
    fn coding_guidance_requires_coding_tools() {
        assert!(build_coding_work_guidance(&["memory_recall"]).is_none());
    }

    #[test]
    fn coding_guidance_mentions_key_workflow_rules() {
        let rendered = build_coding_work_guidance(&[
            "shell",
            "file_read",
            "file_write",
            "file_edit",
            "glob_search",
            "content_search",
            "git_operations",
            "delegate",
        ])
        .expect("coding tools should enable guidance");

        assert!(rendered.contains("## Coding Work"));
        assert!(rendered.contains("Prefer action over questionnaires"));
        assert!(rendered.contains("`file_edit`"));
        assert!(rendered.contains("`shell`"));
        assert!(rendered.contains("`git_operations`"));
        assert!(rendered.contains("`delegate` or `swarm`"));
        assert!(rendered.contains("sensible first version"));
    }
}
