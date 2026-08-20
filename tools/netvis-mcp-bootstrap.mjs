#!/usr/bin/env node
// Bootstrap launcher for the netvis MCP server (Claude Code plugin).
//
// The plugin manifest runs `node` on this script instead of naming the binary,
// because a fresh plugin install is just a checkout: there is no netvis_mcp on
// the machine yet, and plugins have no install-time hook to put one there. The
// launcher closes that gap at first start. Resolution order, first hit wins:
//
//   1. NETVIS_MCP          explicit path to a binary (fails loud if wrong)
//   2. an in-checkout build   build/core or build/release next to this script
//   3. netvis_mcp on PATH     an installer or manual copy
//   4. the download cache     a binary a previous run fetched
//   5. download               the standalone binary asset from the latest
//                             GitHub Release, verified against its SHA256SUMS
//
// Downloads land in CLAUDE_PLUGIN_DATA (survives plugin updates, removed on
// uninstall); outside a plugin context they fall back to ~/.cache/netvis.
// Knobs: NETVIS_MCP_VERSION=X.Y.Z pins the downloaded version,
// NETVIS_MCP_UPDATE=1 re-downloads even with a cached binary, and
// NETVIS_MCP_REPO=owner/name points a fork at its own releases.
// Node stdlib only; no npm install, no extraction tooling (the asset is the
// bare binary).

import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import https from "node:https";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import tls from "node:tls";
import { fileURLToPath } from "node:url";

const REPO = process.env.NETVIS_MCP_REPO || "SeamusMullan/NetVis";
const EXE = process.platform === "win32" ? ".exe" : "";
const BIN = `netvis_mcp${EXE}`;

// Release asset labels keyed by process.platform/arch, matching the names
// package.yml publishes. Combinations outside this table have no published
// binary; the error path lists the ways out.
const LABELS = {
  "linux-x64": "Linux-x86_64",
  "darwin-arm64": "macOS-arm64",
  "win32-x64": "Windows-x64",
};

function fail(msg) {
  process.stderr.write(`netvis-mcp-bootstrap: ${msg}\n`);
  process.exit(1);
}

function note(msg) {
  process.stderr.write(`netvis-mcp-bootstrap: ${msg}\n`);
}

function isRunnable(p) {
  try {
    if (!fs.statSync(p).isFile()) return false;
    // Windows has no execute bit; existence of the .exe is the whole check.
    if (process.platform !== "win32") fs.accessSync(p, fs.constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

// Proxy for a URL per the conventional env vars, or null for a direct
// connection. Node's https module ignores https_proxy on its own, and a
// machine behind a corporate proxy cannot reach GitHub without it.
function proxyFor(url) {
  const e = process.env;
  const proxy = e.https_proxy || e.HTTPS_PROXY || e.http_proxy || e.HTTP_PROXY;
  if (!proxy) return null;
  const host = new URL(url).hostname;
  for (let p of (e.no_proxy || e.NO_PROXY || "").split(",")) {
    p = p.trim();
    if (!p) continue;
    if (p === "*" || host === p || host.endsWith(p.startsWith(".") ? p : `.${p}`)) {
      return null;
    }
  }
  return new URL(proxy);
}

// Open a raw tunnel to host:443 through an HTTP proxy (CONNECT).
function proxyTunnel(proxy, host) {
  return new Promise((resolve, reject) => {
    const headers = { Host: `${host}:443` };
    if (proxy.username) {
      const cred = `${decodeURIComponent(proxy.username)}:${decodeURIComponent(proxy.password)}`;
      headers["Proxy-Authorization"] = `Basic ${Buffer.from(cred).toString("base64")}`;
    }
    const req = http.request({
      host: proxy.hostname,
      port: proxy.port || 3128,
      method: "CONNECT",
      path: `${host}:443`,
      headers,
    });
    req.on("connect", (res, socket) => {
      if (res.statusCode !== 200) {
        socket.destroy();
        reject(new Error(`proxy CONNECT ${host} -> HTTP ${res.statusCode}`));
        return;
      }
      resolve(socket);
    });
    req.on("error", reject);
    req.setTimeout(30_000, () => req.destroy(new Error(`proxy CONNECT ${host} timed out`)));
    req.end();
  });
}

// GET a URL, following redirects (GitHub asset downloads bounce through a
// signed storage URL). Resolves to the response body as a Buffer. Options are
// spelled out rather than passing the URL string: with createConnection set,
// node drops the agent's default port and stamps "Host: <host>:80" onto an
// https request, which GitHub answers with a 400.
async function httpGet(url, redirectsLeft = 5) {
  const u = new URL(url);
  const options = {
    host: u.hostname,
    port: 443,
    path: `${u.pathname}${u.search}`,
    headers: { Host: u.hostname, "User-Agent": "netvis-mcp-bootstrap", Accept: "*/*" },
  };
  const proxy = proxyFor(url);
  if (proxy) {
    const socket = await proxyTunnel(proxy, u.hostname);
    options.createConnection = () => tls.connect({ socket, servername: u.hostname });
  }
  return new Promise((resolve, reject) => {
    const req = https.get(
      options,
      (res) => {
        if ([301, 302, 303, 307, 308].includes(res.statusCode)) {
          res.resume();
          if (!redirectsLeft || !res.headers.location) {
            reject(new Error(`too many redirects for ${url}`));
            return;
          }
          resolve(httpGet(res.headers.location, redirectsLeft - 1));
          return;
        }
        if (res.statusCode !== 200) {
          res.resume();
          reject(new Error(`GET ${url} -> HTTP ${res.statusCode}`));
          return;
        }
        const chunks = [];
        res.on("data", (c) => chunks.push(c));
        res.on("end", () => resolve(Buffer.concat(chunks)));
        res.on("error", reject);
      },
    );
    req.on("error", reject);
    req.setTimeout(60_000, () => req.destroy(new Error(`GET ${url} timed out`)));
  });
}

async function download(dataDir) {
  const key = `${process.platform}-${process.arch}`;
  const label = LABELS[key];
  if (!label) {
    fail(
      `no published netvis_mcp binary for ${key}. Build one from source ` +
        `(cmake --preset core-only && cmake --build --preset core-only) or put ${BIN} on PATH.`,
    );
  }
  const pin = process.env.NETVIS_MCP_VERSION;
  const api = pin
    ? `https://api.github.com/repos/${REPO}/releases/tags/v${pin}`
    : `https://api.github.com/repos/${REPO}/releases/latest`;
  const release = JSON.parse((await httpGet(api)).toString("utf8"));
  const version = String(release.tag_name || "").replace(/^v/, "");
  const assetName = `netvis_mcp-${version}-${label}${EXE}`;
  const assets = release.assets || [];
  const asset = assets.find((a) => a.name === assetName);
  const sums = assets.find((a) => a.name === "SHA256SUMS");
  if (!asset || !sums) {
    fail(
      `release ${release.tag_name} of ${REPO} does not ship ${assetName} and SHA256SUMS ` +
        `(releases up to v0.9.4 predate the standalone binary). ` +
        `Install netvis_mcp on PATH, or pin a newer release with NETVIS_MCP_VERSION.`,
    );
  }

  const [bytes, sumsText] = await Promise.all([
    httpGet(asset.browser_download_url),
    httpGet(sums.browser_download_url),
  ]);

  // SHA256SUMS is standard sha256sum output: "<64 hex>  <name>" per line.
  const expected = new Map(
    sumsText
      .toString("utf8")
      .split("\n")
      .map((l) => l.trim().match(/^([0-9a-f]{64})\s+(.+)$/))
      .filter(Boolean)
      .map((m) => [m[2], m[1]]),
  );
  const got = createHash("sha256").update(bytes).digest("hex");
  if (got !== expected.get(assetName)) {
    fail(
      `checksum mismatch for ${assetName}: SHA256SUMS says ${expected.get(assetName)}, ` +
        `downloaded ${got}. Refusing to run it.`,
    );
  }

  const dest = path.join(dataDir, "bin", BIN);
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  // Write-then-rename so a killed download never leaves a half binary at the
  // path the next run would trust.
  const tmp = `${dest}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, bytes, { mode: 0o755 });
  fs.renameSync(tmp, dest);
  note(`installed netvis_mcp ${version} (${label}) -> ${dest}`);
  return dest;
}

async function resolveBinary() {
  const override = process.env.NETVIS_MCP;
  if (override) {
    if (isRunnable(override)) return override;
    fail(`NETVIS_MCP=${override} is not a runnable file`);
  }

  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  for (const rel of ["build/core", "build/release"]) {
    const p = path.join(root, rel, BIN);
    if (isRunnable(p)) return p;
  }

  for (const dir of (process.env.PATH || "").split(path.delimiter)) {
    if (!dir) continue;
    const p = path.join(dir, BIN);
    if (isRunnable(p)) return p;
  }

  const dataDir =
    process.env.CLAUDE_PLUGIN_DATA || path.join(os.homedir(), ".cache", "netvis");
  const cached = path.join(dataDir, "bin", BIN);
  if (isRunnable(cached) && process.env.NETVIS_MCP_UPDATE !== "1") return cached;

  try {
    return await download(dataDir);
  } catch (err) {
    // An update check that cannot reach the network must not take a working
    // server down; the cached binary keeps serving.
    if (isRunnable(cached)) {
      note(`download failed (${err.message}); using the cached binary`);
      return cached;
    }
    fail(
      `could not download netvis_mcp: ${err.message}. ` +
        `Install it on PATH or point NETVIS_MCP at a binary.`,
    );
  }
}

const bin = await resolveBinary();
const child = spawn(bin, process.argv.slice(2), { stdio: "inherit" });
child.on("error", (err) => fail(`failed to start ${bin}: ${err.message}`));
child.on("exit", (code, signal) => process.exit(signal ? 1 : code ?? 1));
