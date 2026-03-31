#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";

function usage() {
  console.error(
    "Usage: node scripts/openclaw/run-benchmark.mjs --base-url <url> --suite <json> [--out-dir <dir>] [--label <name>]"
  );
  process.exit(1);
}

function getArg(name, fallback = null) {
  const index = process.argv.indexOf(name);
  if (index === -1) {
    return fallback;
  }
  return process.argv[index + 1] ?? fallback;
}

function slugify(value) {
  return String(value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "")
    .slice(0, 60);
}

function nowStamp() {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

function normalizeCases(suite) {
  if (Array.isArray(suite.cases) && suite.cases.length > 0) {
    return suite.cases.map((item, index) => ({
      id: item.id || `case-${index + 1}`,
      title: item.title || item.id || `Case ${index + 1}`,
      intent: item.intent || null,
      steps: Array.isArray(item.steps) ? item.steps : []
    }));
  }
  if (Array.isArray(suite.prompts) && suite.prompts.length > 0) {
    return suite.prompts.map((prompt, index) => ({
      id: prompt.id || `prompt-${index + 1}`,
      title: prompt.title || prompt.id || `Prompt ${index + 1}`,
      intent: prompt.intent || null,
      steps: [
        {
          id: prompt.id || `prompt-${index + 1}`,
          title: prompt.title || prompt.id || `Prompt ${index + 1}`,
          message: prompt.message,
          timeoutMs: prompt.timeoutMs
        }
      ]
    }));
  }
  throw new Error(`Invalid suite: expected non-empty cases or prompts array`);
}

async function fetchJson(url, options = {}, timeoutMs = 120000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    const text = await response.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      data = { raw: text };
    }
    return {
      ok: response.ok,
      status: response.status,
      headers: Object.fromEntries(response.headers.entries()),
      data
    };
  } finally {
    clearTimeout(timeout);
  }
}

async function main() {
  const baseUrl = getArg("--base-url");
  const suitePath = getArg("--suite");
  const outDirArg = getArg("--out-dir", "playground/evals/openclaw");
  const label = getArg("--label", "reference");

  if (!baseUrl || !suitePath) {
    usage();
  }

  const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "../..");
  const suiteFullPath = path.resolve(repoRoot, suitePath);
  const outRoot = path.resolve(repoRoot, outDirArg);

  const suite = JSON.parse(await fs.readFile(suiteFullPath, "utf8"));
  const cases = normalizeCases(suite);

  const runId = `${nowStamp()}-${slugify(label)}`;
  const runDir = path.join(outRoot, runId);
  await fs.mkdir(runDir, { recursive: true });

  const summary = {
    runId,
    label,
    baseUrl,
    suite: path.relative(repoRoot, suiteFullPath),
    startedAt: new Date().toISOString(),
    health: null,
    results: []
  };

  try {
    const health = await fetchJson(new URL("/health", baseUrl), {}, 10000);
    summary.health = health;
    await fs.writeFile(path.join(runDir, "00-health.json"), JSON.stringify(health, null, 2));

    for (let index = 0; index < cases.length; index += 1) {
      const benchmarkCase = cases[index];
      const fileStem = `${String(index + 1).padStart(2, "0")}-${slugify(benchmarkCase.id)}`;
      const steps = [];
      let caseOk = true;
      let totalDurationMs = 0;

      for (let stepIndex = 0; stepIndex < benchmarkCase.steps.length; stepIndex += 1) {
        const step = benchmarkCase.steps[stepIndex];
        const startedAt = Date.now();
        const response = await fetchJson(
          new URL("/prompt", baseUrl),
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: step.message })
          },
          step.timeoutMs || 180000
        );
        const durationMs = Date.now() - startedAt;
        totalDurationMs += durationMs;
        caseOk = caseOk && Boolean(response.ok);
        steps.push({
          id: step.id || `${benchmarkCase.id}-step-${stepIndex + 1}`,
          title: step.title || `Step ${stepIndex + 1}`,
          message: step.message,
          response,
          durationMs
        });
      }

      const record = {
        id: benchmarkCase.id,
        title: benchmarkCase.title,
        intent: benchmarkCase.intent,
        steps,
        totalDurationMs
      };

      await fs.writeFile(path.join(runDir, `${fileStem}.json`), JSON.stringify(record, null, 2));

      const markdown = [
        `# ${record.title}`,
        "",
        `- id: \`${record.id}\``,
        `- totalDurationMs: \`${totalDurationMs}\``,
        `- steps: \`${steps.length}\``,
        `- ok: \`${caseOk}\``,
        "",
        ...steps.flatMap((step, stepIndex) => [
          `## Step ${stepIndex + 1}: ${step.title}`,
          "",
          `- durationMs: \`${step.durationMs}\``,
          `- status: \`${step.response.status}\``,
          `- ok: \`${step.response.ok}\``,
          "",
          "### Prompt",
          "",
          "```text",
          step.message,
          "```",
          "",
          "### Reply",
          "",
          typeof step.response.data?.reply === "string"
            ? step.response.data.reply
            : "```json\n" + JSON.stringify(step.response.data, null, 2) + "\n```",
          ""
        ])
      ].join("\n");

      await fs.writeFile(path.join(runDir, `${fileStem}.md`), markdown);
      summary.results.push({
        id: record.id,
        title: record.title,
        durationMs: totalDurationMs,
        steps: steps.length,
        status: caseOk ? 200 : 500,
        ok: caseOk,
        outputFile: `${fileStem}.md`,
        rawFile: `${fileStem}.json`
      });
    }

    summary.completedAt = new Date().toISOString();
    summary.ok = true;
  } catch (error) {
    summary.completedAt = new Date().toISOString();
    summary.ok = false;
    summary.error = {
      message: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : null
    };
    await fs.writeFile(path.join(runDir, "ERROR.json"), JSON.stringify(summary.error, null, 2));
  }

  await fs.writeFile(path.join(runDir, "summary.json"), JSON.stringify(summary, null, 2));

  const summaryMd = [
    `# OpenClaw Reference Run`,
    "",
    `- runId: \`${summary.runId}\``,
    `- label: \`${summary.label}\``,
    `- baseUrl: \`${summary.baseUrl}\``,
    `- suite: \`${summary.suite}\``,
    `- startedAt: \`${summary.startedAt}\``,
    `- completedAt: \`${summary.completedAt}\``,
    "",
    "## Results",
    "",
    ...summary.results.map(
      (result) =>
        `- \`${result.id}\`: status \`${result.status}\`, duration \`${result.durationMs}ms\`, files \`${result.outputFile}\` / \`${result.rawFile}\``
    ),
    ""
  ].join("\n");
  await fs.writeFile(path.join(runDir, "README.md"), summaryMd);

  console.log(JSON.stringify({ ok: Boolean(summary.ok), runDir }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
