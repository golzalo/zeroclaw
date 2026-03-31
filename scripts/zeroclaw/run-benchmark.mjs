#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { spawn } from "node:child_process";

function usage() {
  console.error(
    "Usage: node scripts/zeroclaw/run-benchmark.mjs --suite <json> [--container <name>] [--out-dir <dir>] [--label <name>]"
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

function stripAnsi(value) {
  return String(value).replace(
    // eslint-disable-next-line no-control-regex
    /[\u001B\u009B][[\]()#;?]*(?:(?:(?:;[-a-zA-Z\d/#&.:=?%@~_]+)*|[\dA-PR-TZcf-nq-uy=><~]))/g,
    ""
  );
}

function normalizeReply(value) {
  const cleaned = stripAnsi(value);
  const lines = cleaned.split(/\r?\n/);
  const filtered = lines.filter((line) => {
    const trimmed = line.trim();
    if (!trimmed) {
      return true;
    }
    if (/^\d+m20\d\d-\d\d-\d\dT/.test(trimmed)) {
      return false;
    }
    if (/^20\d\d-\d\d-\d\dT/.test(trimmed)) {
      return false;
    }
    if (trimmed.includes("zeroclaw::")) {
      return false;
    }
    if (trimmed.startsWith("Config file ")) {
      return false;
    }
    if (trimmed.startsWith("Config loaded ")) {
      return false;
    }
    if (trimmed.startsWith("Memory initialized ")) {
      return false;
    }
    if (trimmed.startsWith("No sandbox backend available")) {
      return false;
    }
    return true;
  });
  return filtered.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

function runCommand(command, args, timeoutMs = 180000) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    let timedOut = false;

    const timeout = setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
    }, timeoutMs);

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });

    child.on("close", (code, signal) => {
      clearTimeout(timeout);
      resolve({
        code,
        signal,
        timedOut,
        stdout: stripAnsi(stdout).trim(),
        stderr: stripAnsi(stderr).trim()
      });
    });
  });
}

async function main() {
  const suitePath = getArg("--suite");
  const container = getArg("--container", "zeroclaw-dev");
  const outDirArg = getArg("--out-dir", "playground/evals/zeroclaw");
  const label = getArg("--label", "zeroclaw-local");

  if (!suitePath) {
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
    container,
    suite: path.relative(repoRoot, suiteFullPath),
    startedAt: new Date().toISOString(),
    results: []
  };

  try {
    for (let index = 0; index < cases.length; index += 1) {
      const benchmarkCase = cases[index];
      const fileStem = `${String(index + 1).padStart(2, "0")}-${slugify(benchmarkCase.id)}`;
      const steps = [];
      let caseOk = true;
      let totalDurationMs = 0;

      for (let stepIndex = 0; stepIndex < benchmarkCase.steps.length; stepIndex += 1) {
        const step = benchmarkCase.steps[stepIndex];
        const startedAt = Date.now();
        const result = await runCommand(
          "docker",
          ["exec", container, "zeroclaw", "agent", "-m", step.message],
          step.timeoutMs || 240000
        );
        const durationMs = Date.now() - startedAt;
        totalDurationMs += durationMs;
        const ok = result.code === 0 && !result.timedOut;
        caseOk = caseOk && ok;
        const response = {
          ok,
          status: ok ? 200 : 500,
          reply: normalizeReply(result.stdout),
          rawStdout: result.stdout,
          stderr: result.stderr,
          exitCode: result.code,
          signal: result.signal,
          timedOut: result.timedOut
        };
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
          step.response.reply ? step.response.reply : "_No stdout reply_",
          "",
          "### Stderr",
          "",
          step.response.stderr ? "```text\n" + step.response.stderr + "\n```" : "_No stderr_",
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

    summary.ok = true;
    summary.completedAt = new Date().toISOString();
  } catch (error) {
    summary.ok = false;
    summary.completedAt = new Date().toISOString();
    summary.error = {
      message: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : null
    };
    await fs.writeFile(path.join(runDir, "ERROR.json"), JSON.stringify(summary.error, null, 2));
  }

  await fs.writeFile(path.join(runDir, "summary.json"), JSON.stringify(summary, null, 2));

  const summaryMd = [
    `# ZeroClaw Benchmark Run`,
    "",
    `- runId: \`${summary.runId}\``,
    `- label: \`${summary.label}\``,
    `- container: \`${summary.container}\``,
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
