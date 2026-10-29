// Minimal HTTP server with live token streaming (SSE) and quant comparison.
// Serves an embedded HTML frontend, zero external HTTP library — raw POSIX sockets.

#include "minillm/model.hpp"
#include "minillm/model_config.hpp"
#include "minillm/ops.hpp"
#include "minillm/safetensors.hpp"
#include "minillm/tokenizer.hpp"

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace minillm;
namespace fs = std::filesystem;
using json   = nlohmann::json;
using clk    = std::chrono::steady_clock;

// ── Embedded HTML frontend ───────────────────────────────────────────────────
static const char* kIndexHtml = R"HTML(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>minillm.cpp · playground</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #0a0a0c;
    --surface: #121216;
    --surface-2: #16161b;
    --border: #22222a;
    --border-strong: #2e2e38;
    --fg: #ededf0;
    --fg-dim: #a3a3ac;
    --muted: #6b6b76;
    --accent: #7c6ff5;        /* electric violet */
    --accent-dim: rgba(124, 111, 245, 0.15);
    --cyan: #38bdf8;
    --amber: #fbbf24;
    --red: #f87171;
    --green: #4ade80;
    --sans: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    --mono: "JetBrains Mono", "SF Mono", ui-monospace, Menlo, monospace;
  }

  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
  @keyframes caret { 0%, 49% { opacity: 1; } 50%, 100% { opacity: 0; } }
  @keyframes tokenIn {
    from { color: var(--accent); background: var(--accent-dim); }
    to   { color: var(--fg); background: transparent; }
  }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(2px); } to { opacity: 1; transform: none; } }

  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; min-height: 100%; }
  body {
    background: var(--bg); color: var(--fg);
    font-family: var(--sans); font-size: 14px; line-height: 1.55;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }
  code, .mono { font-family: var(--mono); font-feature-settings: "ss01", "cv01"; }

  /* ── top nav bar ─────────────────────────────────────────────────────── */
  nav {
    display: flex; align-items: center; gap: 1rem;
    padding: .75rem 1.5rem;
    border-bottom: 1px solid var(--border);
    background: rgba(10, 10, 12, 0.8);
    backdrop-filter: blur(8px);
    position: sticky; top: 0; z-index: 50;
  }
  nav .brand {
    font-weight: 600; letter-spacing: -0.01em;
    display: flex; align-items: center; gap: .5rem;
  }
  nav .brand .dot {
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    background: var(--green); box-shadow: 0 0 8px rgba(74, 222, 128, 0.4);
    animation: pulse 2.5s ease-in-out infinite;
  }
  nav .brand .suffix { color: var(--muted); font-weight: 400; }
  nav .meta { color: var(--muted); font-size: .82rem; font-family: var(--mono); }
  nav .meta .sep { color: var(--border-strong); margin: 0 .5rem; }
  nav .spacer { flex: 1; }
  nav a {
    color: var(--fg-dim); text-decoration: none; font-size: .85rem;
    padding: .25rem .5rem; border-radius: 4px;
  }
  nav a:hover { color: var(--fg); background: var(--surface); }

  main {
    max-width: 920px; margin: 0 auto;
    padding: 2.5rem 1.5rem 4rem;
    animation: fadeIn .4s ease-out;
  }

  /* ── page header ─────────────────────────────────────────────────────── */
  .hero { margin-bottom: 2rem; }
  .hero h1 {
    margin: 0 0 .25rem 0;
    font-size: 1.4rem; font-weight: 600; letter-spacing: -0.02em;
  }
  .hero p { margin: 0; color: var(--fg-dim); font-size: .9rem; }

  /* ── section labels ──────────────────────────────────────────────────── */
  .section-label {
    display: flex; align-items: center; gap: .5rem;
    font-size: .72rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: .5rem;
  }
  .section-label .line {
    flex: 1; height: 1px; background: var(--border);
  }

  /* ── info card ────────────────────────────────────────────────────────── */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.5rem;
  }
  .card.pad-tight { padding: .75rem 1rem; }

  .kv-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: .25rem 1.5rem;
  }
  .kv-grid .kv {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: .3rem 0;
    border-bottom: 1px solid transparent;
  }
  .kv-grid .k { color: var(--muted); font-size: .78rem; }
  .kv-grid .v {
    font-family: var(--mono); font-size: .82rem;
    font-variant-numeric: tabular-nums; color: var(--fg);
  }

  /* ── memory bars ─────────────────────────────────────────────────────── */
  .mem-row {
    display: grid; grid-template-columns: 2.5rem 1fr auto auto;
    align-items: center; gap: .75rem;
    padding: .35rem 0;
  }
  .mem-row:not(:last-child) { border-bottom: 1px solid var(--border); }
  .mem-row .label { color: var(--muted); font-size: .78rem; font-family: var(--mono); }
  .mem-bar {
    height: 6px; background: var(--surface-2); border-radius: 3px; overflow: hidden;
  }
  .mem-bar .fill {
    height: 100%; background: var(--accent);
    transition: width .4s cubic-bezier(.4, 0, .2, 1);
    border-radius: 3px;
  }
  .mem-bar .fill.int8 { background: var(--cyan); }
  .mem-bar .fill.int4 { background: var(--amber); }
  .mem-row .size {
    font-family: var(--mono); font-size: .8rem;
    color: var(--fg); font-variant-numeric: tabular-nums;
    min-width: 4.5rem; text-align: right;
  }
  .mem-row .mult {
    color: var(--muted); font-family: var(--mono); font-size: .72rem;
    min-width: 2rem; text-align: right;
  }

  /* ── prompt input ────────────────────────────────────────────────────── */
  textarea {
    width: 100%; min-height: 68px; resize: vertical;
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    color: var(--fg); padding: .75rem .9rem;
    font-family: var(--mono); font-size: .9rem; line-height: 1.5;
    transition: border-color .15s;
  }
  textarea::placeholder { color: var(--muted); }
  textarea:focus {
    outline: none; border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-dim);
  }

  /* ── controls ────────────────────────────────────────────────────────── */
  .controls {
    display: flex; gap: .75rem; align-items: center; flex-wrap: wrap;
    margin: 1rem 0;
  }
  .field {
    display: inline-flex; align-items: center; gap: .5rem;
    background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
    padding: .35rem .65rem;
  }
  .field label { color: var(--muted); font-size: .78rem; }
  input[type=number], select {
    background: transparent; border: none; color: var(--fg);
    font-family: var(--mono); font-size: .82rem;
    font-variant-numeric: tabular-nums;
  }
  input[type=number] { width: 50px; }
  input:focus, select:focus { outline: none; }
  select { cursor: pointer; }

  .btn {
    background: transparent; border: 1px solid var(--border-strong);
    color: var(--fg); font-family: var(--sans); font-size: .85rem; font-weight: 500;
    padding: .45rem 1rem; border-radius: 6px; cursor: pointer;
    transition: all .15s;
  }
  .btn:hover:not(:disabled) { background: var(--surface); border-color: var(--muted); }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn.primary {
    background: var(--accent); border-color: var(--accent); color: #fff;
  }
  .btn.primary:hover:not(:disabled) {
    background: #8d80ff; border-color: #8d80ff;
    box-shadow: 0 2px 8px rgba(124, 111, 245, 0.3);
  }
  .spacer { flex: 1; }

  .hint { color: var(--muted); font-size: .78rem; margin-left: auto; }
  .hint kbd {
    background: var(--surface); border: 1px solid var(--border);
    border-bottom-width: 2px; border-radius: 3px;
    padding: 1px 5px; font-family: var(--mono); font-size: .72rem;
    color: var(--fg-dim);
  }

  /* ── output ──────────────────────────────────────────────────────────── */
  .output {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 1rem 1.1rem;
    min-height: 180px; white-space: pre-wrap; word-wrap: break-word;
    font-family: var(--mono); font-size: .88rem; line-height: 1.65;
  }
  .output .prompt-txt { color: var(--muted); }
  .output .gen { color: var(--fg); }
  .output .placeholder { color: var(--muted); font-style: italic; }
  .tok-flash { animation: tokenIn .5s ease-out; border-radius: 2px; padding: 0 1px; }
  .caret {
    display: inline-block; width: 7px; height: 1.1em; background: var(--accent);
    vertical-align: text-bottom; margin-left: 1px; animation: caret 1s infinite;
    border-radius: 1px;
  }

  /* ── status bar ──────────────────────────────────────────────────────── */
  .status-bar {
    margin-top: .75rem;
    display: flex; align-items: center; gap: 1.25rem; flex-wrap: wrap;
    padding: .55rem .9rem;
    background: var(--surface-2); border: 1px solid var(--border); border-radius: 6px;
    font-size: .78rem;
  }
  .status-item { display: flex; align-items: center; gap: .4rem; }
  .status-item .k { color: var(--muted); }
  .status-item .v {
    color: var(--fg); font-family: var(--mono);
    font-variant-numeric: tabular-nums;
  }
  .status-item.err .v { color: var(--red); }
  .spark {
    display: inline-flex; align-items: flex-end; height: 16px; gap: 2px;
    margin-left: auto;
  }
  .spark span {
    display: inline-block; width: 3px; background: var(--accent);
    border-radius: 1px; opacity: 0.75;
  }

  /* ── top-K probability panel ─────────────────────────────────────────── */
  .topk {
    margin-top: .75rem;
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: .9rem 1rem;
    display: none;
  }
  .topk.visible { display: block; animation: fadeIn .3s ease-out; }
  .topk .topk-head {
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: .6rem;
  }
  .topk .topk-head .title {
    font-size: .78rem; font-weight: 500; color: var(--fg);
  }
  .topk .topk-head .ctx { color: var(--muted); font-size: .72rem; font-family: var(--mono); }
  .topk-row {
    display: grid; grid-template-columns: 1.5rem minmax(0, 10rem) 1fr 3.5rem;
    gap: .6rem; align-items: center;
    padding: .3rem 0; font-family: var(--mono); font-size: .78rem;
  }
  .topk-row .rank { color: var(--muted); font-size: .7rem; }
  .topk-row .tok {
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    color: var(--fg); background: var(--surface-2); border-radius: 3px;
    padding: 1px 6px;
  }
  .topk-row.chosen .tok {
    color: var(--accent); background: var(--accent-dim);
    box-shadow: inset 0 0 0 1px rgba(124, 111, 245, 0.35);
  }
  .topk-row .bar-track {
    height: 6px; background: var(--surface-2); border-radius: 3px; overflow: hidden;
  }
  .topk-row .bar-fill {
    height: 100%; background: var(--muted); border-radius: 3px;
    transition: width .35s cubic-bezier(.4, 0, .2, 1);
  }
  .topk-row.chosen .bar-fill { background: var(--accent); }
  .topk-row .pct {
    color: var(--fg-dim); font-variant-numeric: tabular-nums;
    text-align: right; font-size: .75rem;
  }
  .topk-row.chosen .pct { color: var(--accent); }

  /* ── compare mode ────────────────────────────────────────────────────── */
  .compare { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1.25rem; }
  .compare .col {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 1rem; display: flex; flex-direction: column; gap: .75rem;
    animation: fadeIn .3s ease-out;
  }
  .compare .col-head {
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: .55rem; border-bottom: 1px solid var(--border);
  }
  .compare .col-head .tag {
    font-family: var(--mono); font-size: .72rem; font-weight: 500;
    padding: 2px 7px; border-radius: 4px;
    background: var(--surface-2);
  }
  .compare .fp32 .tag { color: var(--accent); background: rgba(124, 111, 245, 0.12); }
  .compare .int8 .tag { color: var(--cyan);   background: rgba(56, 189, 248, 0.12); }
  .compare .int4 .tag { color: var(--amber);  background: rgba(251, 191, 36, 0.12); }
  .compare .perf {
    font-family: var(--mono); font-size: .74rem; color: var(--fg-dim);
    font-variant-numeric: tabular-nums;
  }
  .compare .perf .num { color: var(--fg); }
  .compare .text {
    font-family: var(--mono); font-size: .78rem; line-height: 1.6;
    white-space: pre-wrap; word-wrap: break-word;
    min-height: 120px; max-height: 300px; overflow-y: auto;
    color: var(--fg);
  }
  .compare .text .prompt-txt { color: var(--muted); }

  @media (max-width: 780px) {
    .kv-grid { grid-template-columns: repeat(2, 1fr); }
    .compare { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<nav>
  <span class="brand">
    <span class="dot"></span>
    <span>minillm<span class="suffix">.cpp</span></span>
  </span>
  <span class="meta">
    SmolLM2-135M<span class="sep">·</span>ARM NEON<span class="sep">·</span>:8080
  </span>
  <span class="spacer"></span>
  <a href="https://github.com/georgebon5/minillm-cpp" target="_blank">source</a>
</nav>

<main>
  <div class="hero">
    <h1>Inference playground</h1>
    <p>Run the CPU-only inference engine and compare generation quality across quantization levels.</p>
  </div>

  <!-- model section -->
  <div class="section-label">Model <div class="line"></div></div>
  <div class="card">
    <div class="kv-grid">
      <div class="kv"><span class="k">vocab</span><span class="v" id="iVocab">—</span></div>
      <div class="kv"><span class="k">layers</span><span class="v" id="iLayers">—</span></div>
      <div class="kv"><span class="k">hidden</span><span class="v" id="iHidden">—</span></div>
      <div class="kv"><span class="k">heads</span><span class="v" id="iHeads">—</span></div>
      <div class="kv"><span class="k">kv-heads</span><span class="v" id="iKvHeads">—</span></div>
      <div class="kv"><span class="k">head-dim</span><span class="v" id="iHeadDim">—</span></div>
    </div>
  </div>

  <!-- memory section -->
  <div class="section-label">Memory footprint <div class="line"></div></div>
  <div class="card pad-tight">
    <div class="mem-row">
      <span class="label">fp32</span>
      <div class="mem-bar"><div class="fill" id="barFp32" style="width:100%"></div></div>
      <span class="size" id="sizeFp32">—</span>
      <span class="mult">1×</span>
    </div>
    <div class="mem-row">
      <span class="label">int8</span>
      <div class="mem-bar"><div class="fill int8" id="barInt8" style="width:0"></div></div>
      <span class="size" id="sizeInt8">—</span>
      <span class="mult" id="multInt8">—</span>
    </div>
    <div class="mem-row">
      <span class="label">int4</span>
      <div class="mem-bar"><div class="fill int4" id="barInt4" style="width:0"></div></div>
      <span class="size" id="sizeInt4">—</span>
      <span class="mult" id="multInt4">—</span>
    </div>
  </div>

  <!-- prompt section -->
  <div class="section-label">Prompt <div class="line"></div></div>
  <textarea id="prompt" placeholder="enter prompt and press ⌘⏎">The capital of France is</textarea>

  <div class="controls">
    <div class="field"><label>tokens</label>
      <input id="maxTokens" type="number" value="48" min="1" max="256">
    </div>
    <div class="field"><label>quant</label>
      <select id="quant">
        <option value="fp32" selected>fp32</option>
        <option value="int8">int8</option>
        <option value="int4">int4</option>
      </select>
    </div>
    <div class="field">
      <label>temp</label>
      <input id="temperature" type="range" min="0" max="1.5" step="0.1" value="0.8"
             style="width:80px;accent-color:var(--accent);">
      <span id="tempVal" class="mono" style="font-size:.78rem;color:var(--fg);min-width:1.8rem;">0.8</span>
    </div>
    <span class="hint"><kbd>⌘⏎</kbd> run · <kbd>esc</kbd> stop</span>
    <button class="btn" id="compareBtn">Compare all</button>
    <button class="btn primary" id="goBtn">Generate</button>
  </div>

  <!-- output section -->
  <div class="section-label">Output <div class="line"></div></div>
  <div class="output" id="out"><span class="placeholder">Output will appear here.</span></div>

  <div class="status-bar">
    <div class="status-item"><span class="k">tokens</span><span class="v" id="statN">0</span></div>
    <div class="status-item"><span class="k">elapsed</span><span class="v" id="statMs">0ms</span></div>
    <div class="status-item"><span class="k">throughput</span><span class="v" id="statTps">0 tok/s</span></div>
    <div class="status-item"><span class="k">quant</span><span class="v" id="statQ">—</span></div>
    <div class="spark" id="spark"></div>
  </div>

  <div class="topk" id="topk">
    <div class="topk-head">
      <span class="title">Top candidates at step <span id="topkStep">—</span></span>
      <span class="ctx">top-5 softmax probabilities</span>
    </div>
    <div id="topkRows"></div>
  </div>

  <div class="compare" id="compare" style="display:none"></div>
</main>

<script>
const $ = (id) => document.getElementById(id);
const promptEl = $('prompt');
const outEl    = $('out');
const compareEl = $('compare');
const goBtn   = $('goBtn');
const compareBtn = $('compareBtn');
const spark   = $('spark');
const tempEl  = $('temperature');
const tempValEl = $('tempVal');

tempEl.addEventListener('input', () => {
  tempValEl.textContent = parseFloat(tempEl.value).toFixed(1);
});

let activeStream = null;
let tpsHistory = [];

// ── model info ──────────────────────────────────────────────────────────────
async function loadInfo() {
  try {
    const r = await fetch('/info');
    const d = await r.json();
    $('iVocab').textContent   = d.vocab.toLocaleString();
    $('iLayers').textContent  = d.layers;
    $('iHidden').textContent  = d.hidden;
    $('iHeads').textContent   = d.n_heads;
    $('iKvHeads').textContent = d.n_kv_heads + ' (GQA)';
    $('iHeadDim').textContent = d.head_dim;
    $('sizeFp32').textContent = fmtMB(d.mem_fp32_bytes);
    $('sizeInt8').textContent = fmtMB(d.mem_int8_bytes);
    $('sizeInt4').textContent = fmtMB(d.mem_int4_bytes);
    $('barInt8').style.width = (100 * d.mem_int8_bytes / d.mem_fp32_bytes) + '%';
    $('barInt4').style.width = (100 * d.mem_int4_bytes / d.mem_fp32_bytes) + '%';
    $('multInt8').textContent = (d.mem_fp32_bytes/d.mem_int8_bytes).toFixed(1) + '×';
    $('multInt4').textContent = (d.mem_fp32_bytes/d.mem_int4_bytes).toFixed(1) + '×';
  } catch (e) { console.warn('info failed', e); }
}
function fmtMB(b) { return (b/1024/1024).toFixed(1) + ' MB'; }

// ── sparkline ───────────────────────────────────────────────────────────────
function pushSpark(tps) {
  tpsHistory.push(tps);
  if (tpsHistory.length > 32) tpsHistory.shift();
  const max = Math.max(1, ...tpsHistory);
  spark.innerHTML = '';
  for (const v of tpsHistory) {
    const s = document.createElement('span');
    s.style.height = Math.max(2, 16 * v / max) + 'px';
    spark.appendChild(s);
  }
}

// ── streaming ───────────────────────────────────────────────────────────────
function streamGenerate() {
  stopStream();
  compareEl.style.display = 'none';

  const prompt     = promptEl.value;
  const max_tokens = parseInt($('maxTokens').value, 10);
  const quant      = $('quant').value;

  if (!prompt.trim()) return;

  goBtn.disabled = true;
  tpsHistory = []; spark.innerHTML = '';
  $('statN').textContent  = '0';
  $('statMs').textContent = '0ms';
  $('statTps').textContent = '0 tok/s';
  $('statQ').textContent  = quant;

  outEl.innerHTML = '<span class="prompt-txt"></span><span class="gen"></span><span class="caret"></span>';
  outEl.querySelector('.prompt-txt').textContent = prompt;
  const genSpan = outEl.querySelector('.gen');

  const temperature = parseFloat(tempEl.value);
  const url = `/stream?prompt=${encodeURIComponent(prompt)}&max_tokens=${max_tokens}&quant=${quant}&temperature=${temperature}`;
  const es  = new EventSource(url);
  activeStream = es;

  const t0 = performance.now();

  es.addEventListener('token', (e) => {
    const d = JSON.parse(e.data);
    const flash = document.createElement('span');
    flash.className = 'tok-flash';
    flash.textContent = d.delta;
    genSpan.appendChild(flash);
    setTimeout(() => {
      const t = flash.textContent;
      flash.replaceWith(document.createTextNode(t));
    }, 500);

    if (d.topk) renderTopK(d.topk, d.id, d.n);

    const elapsed = (performance.now() - t0) / 1000;
    const tps = d.n / elapsed;
    pushSpark(tps);
    $('statN').textContent  = d.n;
    $('statMs').textContent = Math.round(elapsed * 1000) + 'ms';
    $('statTps').textContent = tps.toFixed(0) + ' tok/s';
  });

  es.addEventListener('done', (e) => {
    const d = JSON.parse(e.data);
    outEl.querySelector('.caret')?.remove();
    $('statN').textContent  = d.n_tokens;
    $('statMs').textContent = d.ms + 'ms';
    $('statTps').textContent = d.tok_per_sec + ' tok/s';
    es.close(); activeStream = null; goBtn.disabled = false;
  });

  es.onerror = () => {
    outEl.querySelector('.caret')?.remove();
    $('statMs').textContent = 'connection lost';
    es.close(); activeStream = null; goBtn.disabled = false;
  };
}

function stopStream() {
  if (activeStream) { activeStream.close(); activeStream = null; }
  outEl.querySelector('.caret')?.remove();
  goBtn.disabled = false;
}

// ── compare ─────────────────────────────────────────────────────────────────
async function compareAll() {
  stopStream();
  const prompt     = promptEl.value;
  const max_tokens = parseInt($('maxTokens').value, 10);
  if (!prompt.trim()) return;

  compareBtn.disabled = true; goBtn.disabled = true;
  outEl.innerHTML = '<span class="placeholder">Running fp32, int8, and int4 in sequence…</span>';
  compareEl.innerHTML = '';
  compareEl.style.display = 'grid';
  $('statQ').textContent = 'all';

  try {
    const r = await fetch('/compare', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ prompt, max_tokens, temperature: parseFloat(tempEl.value) }),
    });
    const data = await r.json();
    compareEl.innerHTML = '';
    for (const q of ['fp32', 'int8', 'int4']) {
      const d = data[q];
      const col = document.createElement('div');
      col.className = 'col ' + q;
      col.innerHTML = `
        <div class="col-head">
          <span class="tag">${q}</span>
          <span class="perf"><span class="num">${d.tok_per_sec}</span> tok/s</span>
        </div>
        <div class="text"><span class="prompt-txt">${escapeHtml(prompt)}</span>${escapeHtml(d.generated)}</div>
        <div class="perf"><span class="num">${d.n_tokens}</span> tokens · <span class="num">${d.ms}</span>ms</div>
      `;
      compareEl.appendChild(col);
    }
    outEl.innerHTML = '<span class="placeholder">Comparison below — same prompt across three quantization levels.</span>';
  } catch (e) {
    outEl.innerHTML = `<span class="placeholder" style="color: var(--red)">Error: ${escapeHtml(e.message)}</span>`;
  } finally {
    compareBtn.disabled = false; goBtn.disabled = false;
  }
}

// Render top-K candidates as a bar chart.
function renderTopK(items, chosenId, step) {
  const panel = $('topk');
  const rows  = $('topkRows');
  panel.classList.add('visible');
  $('topkStep').textContent = step;

  const maxP = Math.max(...items.map(x => x.p));
  rows.innerHTML = items.map((x, i) => {
    const picked = x.id === chosenId;
    const w = (x.p / maxP * 100).toFixed(1);
    const label = x.token.replace(/\n/g, '␤').replace(/\t/g, '→').replace(/ /g, '·');
    return `
      <div class="topk-row ${picked ? 'chosen' : ''}">
        <span class="rank">#${i+1}</span>
        <span class="tok">${escapeHtml(label)}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${w}%"></div></div>
        <span class="pct">${(x.p*100).toFixed(1)}%</span>
      </div>`;
  }).join('');
}

function escapeHtml(s) {
  return (s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

goBtn.addEventListener('click', streamGenerate);
compareBtn.addEventListener('click', compareAll);
promptEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) streamGenerate();
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') stopStream();
});

loadInfo();
</script>
)HTML";

// ── Globals set once at startup ──────────────────────────────────────────────
struct ServerState {
    ModelConfig  cfg;
    ModelWeights w_fp32;
    ModelWeights w_int8;
    ModelWeights w_int4;
    bool         int8_ready = false;
    bool         int4_ready = false;
    RopeCache    rope;
    Tokenizer    tok;

    size_t mem_fp32_bytes = 0;
    size_t mem_int8_bytes = 0;  // estimated
    size_t mem_int4_bytes = 0;  // estimated
};

static const ModelWeights& ensure_weights(ServerState& st, const std::string& quant) {
    if (quant == "int8") {
        if (!st.int8_ready) {
            std::cerr << "Quantizing to INT8 (one-time)...\n";
            st.w_int8     = quantize_weights(st.w_fp32, DType::I8);
            st.int8_ready = true;
        }
        return st.w_int8;
    }
    if (quant == "int4") {
        if (!st.int4_ready) {
            std::cerr << "Quantizing to INT4 (one-time)...\n";
            st.w_int4     = quantize_weights(st.w_fp32, DType::I4);
            st.int4_ready = true;
        }
        return st.w_int4;
    }
    return st.w_fp32;
}

// ── Byte accounting ──────────────────────────────────────────────────────────

static size_t weight_bytes(const ModelWeights& w) {
    size_t b = w.embed.nbytes() + w.final_norm.nbytes() + w.lm_head.nbytes();
    for (const auto& l : w.layers)
        b += l.attn_norm.nbytes() + l.Wq.nbytes() + l.Wk.nbytes() + l.Wv.nbytes() +
             l.Wo.nbytes() + l.mlp_norm.nbytes() + l.W_gate.nbytes() + l.W_up.nbytes() +
             l.W_down.nbytes();
    return b;
}

// Length of the longest valid UTF-8 prefix of s. A trailing incomplete
// multi-byte sequence is excluded so JSON serialization never fails and
// SSE token deltas don't emit partial codepoints.
static size_t utf8_safe_len(const std::string& s) {
    const size_t n = s.size();
    if (n == 0) return 0;
    // Walk back from end; find the start byte of the last codepoint.
    size_t i = n;
    while (i > 0) {
        --i;
        uint8_t c = static_cast<uint8_t>(s[i]);
        if ((c & 0x80) == 0) return i + 1;        // ASCII, complete
        if ((c & 0xC0) == 0x80) continue;         // continuation byte, keep going
        // start byte: how many total bytes does this codepoint need?
        size_t need = 0;
        if      ((c & 0xE0) == 0xC0) need = 2;
        else if ((c & 0xF0) == 0xE0) need = 3;
        else if ((c & 0xF8) == 0xF0) need = 4;
        else return i;                            // malformed, drop byte
        return (n - i >= need) ? n : i;           // complete at tail or truncate
    }
    return 0;
}

// Serialize JSON with invalid UTF-8 replaced by U+FFFD (never throw).
static std::string safe_dump(const json& j) {
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

// ── HTTP helpers ─────────────────────────────────────────────────────────────

static std::string http_response(int status, const std::string& ctype,
                                  const std::string& body) {
    const char* st = (status == 200) ? "OK" : (status == 400) ? "Bad Request"
                   : (status == 404) ? "Not Found" : "Internal Server Error";
    std::string r;
    r += "HTTP/1.1 " + std::to_string(status) + " " + st + "\r\n";
    r += "Content-Type: " + ctype + "\r\n";
    r += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    r += "Connection: close\r\n\r\n";
    r += body;
    return r;
}

static void send_all(int sock, const std::string& data) {
    size_t sent = 0;
    while (sent < data.size()) {
        ssize_t n = send(sock, data.data() + sent, data.size() - sent, 0);
        if (n <= 0) return;
        sent += static_cast<size_t>(n);
    }
}

static std::string read_request(int sock) {
    std::string buf;
    char        chunk[4096];
    size_t      header_end = std::string::npos;
    while (header_end == std::string::npos) {
        ssize_t n = recv(sock, chunk, sizeof(chunk), 0);
        if (n <= 0) return {};
        buf.append(chunk, static_cast<size_t>(n));
        header_end = buf.find("\r\n\r\n");
    }
    size_t content_len = 0;
    const std::string key = "Content-Length:";
    size_t pos = buf.find(key);
    if (pos != std::string::npos && pos < header_end) {
        size_t eol = buf.find("\r\n", pos);
        content_len = static_cast<size_t>(std::stoul(buf.substr(pos + key.size(), eol - pos - key.size())));
    }
    size_t body_start = header_end + 4;
    while (buf.size() - body_start < content_len) {
        ssize_t n = recv(sock, chunk, sizeof(chunk), 0);
        if (n <= 0) break;
        buf.append(chunk, static_cast<size_t>(n));
    }
    return buf;
}

// Minimal URL-decode (for query string values).
static std::string url_decode(const std::string& s) {
    std::string out; out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '+') out += ' ';
        else if (s[i] == '%' && i + 2 < s.size()) {
            int hi = std::stoi(s.substr(i+1, 2), nullptr, 16);
            out += static_cast<char>(hi);
            i += 2;
        } else out += s[i];
    }
    return out;
}

static std::string query_param(const std::string& qs, const std::string& key) {
    std::string prefix = key + "=";
    size_t pos = qs.find(prefix);
    if (pos == std::string::npos) return "";
    pos += prefix.size();
    size_t end = qs.find('&', pos);
    return url_decode(qs.substr(pos, end - pos));
}

// ── Shared generation (used by both /generate and /compare) ──────────────────

struct GenResult {
    std::string generated;
    int         n_tokens;
    int         ms;
    int         tok_per_sec;
};

static GenResult run_generate(ServerState& st, const ModelWeights& w,
                               const std::string& prompt, int max_tokens,
                               const SamplingParams& sampling) {
    auto prompt_ids = encode(st.tok, prompt);
    auto t0 = clk::now();
    auto ids = generate(w, st.cfg, st.rope, prompt_ids, max_tokens, st.tok.eos_id, sampling);
    int ms = static_cast<int>(std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    double tps = ids.empty() ? 0 : static_cast<double>(ids.size()) / (ms / 1000.0);
    return { decode(st.tok, ids), static_cast<int>(ids.size()), ms, static_cast<int>(tps) };
}

// Parse sampling params from a JSON body or query string helpers.
static SamplingParams parse_sampling(const json& req) {
    SamplingParams sp;
    sp.temperature = std::max(0.0f, std::min(2.0f, req.value("temperature", 0.0f)));
    sp.top_k       = std::max(1, std::min(200, req.value("top_k", 40)));
    sp.seed        = req.value("seed", static_cast<uint64_t>(0));
    return sp;
}

// ── Endpoint handlers ────────────────────────────────────────────────────────

static std::string handle_info(const ServerState& st) {
    json j = {
        {"vocab",      st.cfg.vocab_size},
        {"layers",     st.cfg.num_hidden_layers},
        {"hidden",     st.cfg.hidden_size},
        {"n_heads",    st.cfg.num_attention_heads},
        {"n_kv_heads", st.cfg.num_key_value_heads},
        {"head_dim",   st.cfg.head_dim},
        {"max_pos",    st.cfg.max_position_embeddings},
        {"mem_fp32_bytes", st.mem_fp32_bytes},
        {"mem_int8_bytes", st.mem_int8_bytes},
        {"mem_int4_bytes", st.mem_int4_bytes},
    };
    return http_response(200, "application/json", safe_dump(j));
}

static std::string handle_generate(ServerState& st, const std::string& body) {
    json req;
    try { req = json::parse(body); }
    catch (const std::exception& e) {
        return http_response(400, "application/json",
                             json{{"error", std::string("bad json: ") + e.what()}}.dump());
    }
    std::string prompt     = req.value("prompt", "");
    int         max_tokens = std::max(1, std::min(256, req.value("max_tokens", 48)));
    std::string quant      = req.value("quant", "fp32");
    if (prompt.empty())
        return http_response(400, "application/json", R"({"error":"empty prompt"})");

    const ModelWeights& w = ensure_weights(st, quant);
    auto r = run_generate(st, w, prompt, max_tokens, parse_sampling(req));
    json j = {{"prompt",prompt},{"generated",r.generated},{"n_tokens",r.n_tokens},
              {"ms",r.ms},{"tok_per_sec",r.tok_per_sec}};
    return http_response(200, "application/json", safe_dump(j));
}

static std::string handle_compare(ServerState& st, const std::string& body) {
    json req;
    try { req = json::parse(body); }
    catch (const std::exception& e) {
        return http_response(400, "application/json",
                             json{{"error", std::string("bad json: ") + e.what()}}.dump());
    }
    std::string prompt     = req.value("prompt", "");
    int         max_tokens = std::max(1, std::min(256, req.value("max_tokens", 48)));
    if (prompt.empty())
        return http_response(400, "application/json", R"({"error":"empty prompt"})");

    SamplingParams base_sp = parse_sampling(req);
    // Use a fixed seed across all three so the same tokens are compared.
    if (base_sp.seed == 0) base_sp.seed = static_cast<uint64_t>(std::random_device{}());

    json result;
    for (const char* q : {"fp32", "int8", "int4"}) {
        const ModelWeights& w = ensure_weights(st, q);
        auto r = run_generate(st, w, prompt, max_tokens, base_sp);
        result[q] = {{"generated",r.generated},{"n_tokens",r.n_tokens},
                     {"ms",r.ms},{"tok_per_sec",r.tok_per_sec}};
    }
    return http_response(200, "application/json", safe_dump(result));
}

// SSE: inline the generate loop so we can push one token at a time.
static void handle_stream(ServerState& st, int sock, const std::string& qs) {
    std::string prompt     = query_param(qs, "prompt");
    int         max_tokens = std::max(1, std::min(256, std::atoi(query_param(qs, "max_tokens").c_str())));
    std::string quant      = query_param(qs, "quant");
    if (quant.empty()) quant = "fp32";

    SamplingParams sp;
    std::string t_str = query_param(qs, "temperature");
    std::string k_str = query_param(qs, "top_k");
    if (!t_str.empty()) sp.temperature = std::max(0.0f, std::min(2.0f, std::stof(t_str)));
    if (!k_str.empty()) sp.top_k       = std::max(1, std::min(200, std::atoi(k_str.c_str())));

    if (prompt.empty()) {
        send_all(sock, http_response(400, "text/plain", "empty prompt"));
        return;
    }

    // SSE headers
    std::string hdr = "HTTP/1.1 200 OK\r\n"
                      "Content-Type: text/event-stream\r\n"
                      "Cache-Control: no-cache\r\n"
                      "Connection: close\r\n"
                      "X-Accel-Buffering: no\r\n\r\n";
    send_all(sock, hdr);

    auto emit = [sock](const std::string& event, const std::string& data) {
        std::string msg = "event: " + event + "\ndata: " + data + "\n\n";
        send_all(sock, msg);
    };

    const ModelWeights& w = ensure_weights(st, quant);

    auto prompt_ids = encode(st.tok, prompt);
    if (prompt_ids.empty()) {
        emit("done", R"({"n_tokens":0,"ms":0,"tok_per_sec":0})");
        return;
    }

    KVCache kv = make_kvcache(st.cfg);
    Tensor logits;
    for (int32_t id : prompt_ids)
        logits = forward_one(w, st.cfg, st.rope, id, kv);

    std::vector<int32_t> gen_ids;
    std::string prev_text;
    auto t0 = clk::now();
    const size_t vocab = static_cast<size_t>(st.cfg.vocab_size);

    std::mt19937 rng(std::random_device{}());

    auto sample = [&](const float* lp) -> int32_t {
        if (sp.temperature <= 0.0f) {
            int32_t best = 0; float best_val = lp[0];
            for (size_t v = 1; v < vocab; ++v)
                if (lp[v] > best_val) { best_val = lp[v]; best = static_cast<int32_t>(v); }
            return best;
        }
        const size_t k   = std::min<size_t>(static_cast<size_t>(sp.top_k), vocab);
        const float  inv = 1.0f / sp.temperature;
        std::vector<std::pair<float, int32_t>> scored(vocab);
        for (size_t v = 0; v < vocab; ++v)
            scored[v] = { lp[v] * inv, static_cast<int32_t>(v) };
        std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                          [](auto& a, auto& b){ return a.first > b.first; });
        float max_v = scored[0].first; float sum = 0.f;
        std::vector<float> probs(k);
        for (size_t i = 0; i < k; ++i) { probs[i] = std::exp(scored[i].first - max_v); sum += probs[i]; }
        for (auto& p : probs) p /= sum;
        std::uniform_real_distribution<float> u(0.f, 1.f);
        float r = u(rng), acc = 0.f;
        for (size_t i = 0; i < k; ++i) { acc += probs[i]; if (r < acc) return scored[i].second; }
        return scored[k - 1].second;
    };

    // Compute top-K softmax probabilities + decoded strings from logits.
    // Used by the UI to show the model's "thought process" per step.
    auto topk_probs = [&](const float* lp, int K = 5) -> json {
        std::vector<std::pair<float, int32_t>> scored(vocab);
        for (size_t v = 0; v < vocab; ++v) scored[v] = { lp[v], static_cast<int32_t>(v) };
        std::partial_sort(scored.begin(), scored.begin() + K, scored.end(),
                          [](auto& a, auto& b){ return a.first > b.first; });
        float max_v = scored[0].first; float sum = 0.f;
        std::vector<float> probs(K);
        for (int i = 0; i < K; ++i) { probs[i] = std::exp(scored[i].first - max_v); sum += probs[i]; }
        for (auto& p : probs) p /= sum;

        json arr = json::array();
        for (int i = 0; i < K; ++i) {
            std::vector<int32_t> one = { scored[i].second };
            std::string s = decode(st.tok, one);
            // Replace invalid UTF-8 fragments so JSON serializes cleanly.
            if (utf8_safe_len(s) < s.size()) s = s.substr(0, utf8_safe_len(s));
            arr.push_back({ {"token", s}, {"id", scored[i].second}, {"p", probs[i]} });
        }
        return arr;
    };

    for (int step = 0; step < max_tokens; ++step) {
        const float* lp = logits.data_as<float>();
        json candidates = topk_probs(lp, 5);
        int32_t next_id = sample(lp);

        gen_ids.push_back(next_id);

        // Decode cumulative, trim trailing incomplete UTF-8, emit only the
        // completed suffix. Byte-level BPE may split a codepoint across tokens.
        std::string full = decode(st.tok, gen_ids);
        size_t safe = utf8_safe_len(full);
        if (safe > prev_text.size()) {
            std::string delta = full.substr(prev_text.size(), safe - prev_text.size());
            prev_text = full.substr(0, safe);
            json ev = {{"delta",delta},{"id",next_id},{"n",gen_ids.size()},
                       {"topk", candidates}};
            emit("token", safe_dump(ev));
        }

        if (next_id == st.tok.eos_id) break;
        logits = forward_one(w, st.cfg, st.rope, next_id, kv);
    }

    // Flush any remainder (should be rare — final tokens usually complete all UTF-8).
    std::string final_full = decode(st.tok, gen_ids);
    if (final_full.size() > prev_text.size()) {
        std::string tail = final_full.substr(prev_text.size());
        json ev = {{"delta",tail},{"id",-1},{"n",gen_ids.size()}};
        emit("token", safe_dump(ev));
    }

    int ms = static_cast<int>(std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    double tps = gen_ids.empty() ? 0 : static_cast<double>(gen_ids.size()) / (ms / 1000.0);
    json done = {{"n_tokens",gen_ids.size()},{"ms",ms},{"tok_per_sec",static_cast<int>(tps)}};
    emit("done", safe_dump(done));
}

// ── Dispatcher ───────────────────────────────────────────────────────────────

static void dispatch(ServerState& st, int sock, const std::string& raw) {
    size_t eol = raw.find("\r\n");
    if (eol == std::string::npos) { send_all(sock, http_response(400, "text/plain", "bad")); return; }
    std::string line = raw.substr(0, eol);
    size_t sp1 = line.find(' ');
    size_t sp2 = line.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) {
        send_all(sock, http_response(400, "text/plain", "bad")); return;
    }
    std::string method = line.substr(0, sp1);
    std::string full   = line.substr(sp1 + 1, sp2 - sp1 - 1);
    std::string path = full, query;
    size_t qp = full.find('?');
    if (qp != std::string::npos) { path = full.substr(0, qp); query = full.substr(qp + 1); }

    if (method == "GET" && (path == "/" || path == "/index.html"))
        return send_all(sock, http_response(200, "text/html; charset=utf-8", kIndexHtml));
    if (method == "GET" && path == "/info")
        return send_all(sock, handle_info(st));
    if (method == "GET" && path == "/stream")
        return handle_stream(st, sock, query);

    size_t header_end = raw.find("\r\n\r\n");
    std::string body  = (header_end != std::string::npos) ? raw.substr(header_end + 4) : "";

    if (method == "POST" && path == "/generate")
        return send_all(sock, handle_generate(st, body));
    if (method == "POST" && path == "/compare")
        return send_all(sock, handle_compare(st, body));

    send_all(sock, http_response(404, "text/plain", "not found"));
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string model_dir = "";
    int         port      = 8080;
    for (int i = 1; i < argc; ++i) {
        if      (std::strcmp(argv[i], "--help")  == 0) {
            std::cerr << "Usage: " << argv[0] << " --model <dir> [--port 8080]\n";
            return 0;
        }
        else if (std::strcmp(argv[i], "--model") == 0 && i+1 < argc) model_dir = argv[++i];
        else if (std::strcmp(argv[i], "--port")  == 0 && i+1 < argc) port = std::atoi(argv[++i]);
    }
    if (model_dir.empty()) {
        std::cerr << "Usage: " << argv[0] << " --model <dir> [--port 8080]\n";
        return 1;
    }

    const fs::path dir(model_dir);
    for (const auto& p : {dir/"config.json", dir/"model.safetensors", dir/"tokenizer.json"}) {
        if (!fs::exists(p)) { std::cerr << "Missing: " << p << "\n"; return 1; }
    }

    std::cerr << "Loading model...\n";
    ServerState st;
    st.cfg    = load_config((dir/"config.json").string());
    SafetensorsFile sf((dir/"model.safetensors").string());
    st.w_fp32 = load_weights(sf, st.cfg);
    st.rope   = build_rope_cache(st.cfg.max_position_embeddings,
                                  st.cfg.head_dim, st.cfg.rope_theta);
    st.tok    = load_tokenizer((dir/"tokenizer.json").string());

    // Memory estimates (projection matrices scale, embed/norms kept f32)
    st.mem_fp32_bytes = weight_bytes(st.w_fp32);
    // Rough estimate: proj matrices ~ 90% of weights; int8 = /4, int4 = /8
    size_t proj_f32 = st.mem_fp32_bytes - st.w_fp32.embed.nbytes()
                                        - st.w_fp32.final_norm.nbytes();
    st.mem_int8_bytes = st.w_fp32.embed.nbytes() + st.w_fp32.final_norm.nbytes() + proj_f32 / 4;
    st.mem_int4_bytes = st.w_fp32.embed.nbytes() + st.w_fp32.final_norm.nbytes() + proj_f32 / 8;

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { perror("socket"); return 1; }
    int yes = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = htons(static_cast<uint16_t>(port));
    if (bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        perror("bind"); return 1;
    }
    if (listen(srv, 8) < 0) { perror("listen"); return 1; }

    std::cerr << "\nminillm server ready → http://localhost:" << port << "\n";
    std::cerr << "Press Ctrl+C to stop.\n\n";

    while (true) {
        sockaddr_in cli{};
        socklen_t   cli_len = sizeof(cli);
        int sock = accept(srv, reinterpret_cast<sockaddr*>(&cli), &cli_len);
        if (sock < 0) { perror("accept"); continue; }
        std::string raw = read_request(sock);
        if (!raw.empty()) dispatch(st, sock, raw);
        close(sock);
    }
    return 0;
}
