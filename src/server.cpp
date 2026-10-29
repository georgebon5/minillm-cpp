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
<title>MINILLM.CPP · ONLINE</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #050807;
    --bg2: #0a0f0d;
    --panel: #0d1512;
    --border: #163026;
    --border-hot: #1f5944;
    --fg: #c7f5db;
    --dim: #4a7a62;
    --mute: #2d4a3d;
    --phos: #00ff88;
    --cyan: #22e8ff;
    --amber: #ffb84d;
    --red: #ff3344;
    --mono: "JetBrains Mono", "SF Mono", ui-monospace, Menlo, monospace;
  }

  @keyframes scan { 0% { transform: translateY(-100%); } 100% { transform: translateY(100vh); } }
  @keyframes flicker { 0%, 100% { opacity: 1; } 50% { opacity: .97; } }
  @keyframes pulse-dot { 0%, 100% { opacity: 1; } 50% { opacity: .3; } }
  @keyframes caret-blink { 0%, 49% { opacity: 1; } 50%, 100% { opacity: 0; } }
  @keyframes token-flash {
    0% { background: rgba(0, 255, 136, 0.25); color: var(--phos); }
    100% { background: transparent; color: var(--fg); }
  }
  @keyframes boot-type { from { width: 0; } to { width: 100%; } }

  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--fg); min-height: 100%; }
  body {
    font-family: var(--mono);
    font-size: 13px; line-height: 1.55;
    animation: flicker 7s infinite;
    position: relative; overflow-x: hidden;
    background:
      radial-gradient(ellipse at top, #0a1510 0%, var(--bg) 60%),
      repeating-linear-gradient(0deg,
        rgba(0, 255, 136, 0.028) 0px,
        rgba(0, 255, 136, 0.028) 1px,
        transparent 1px,
        transparent 3px);
  }
  /* moving scanline */
  body::before {
    content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 100;
    background: linear-gradient(180deg, transparent 0%, rgba(0, 255, 136, 0.08) 50%, transparent 100%);
    height: 100px; animation: scan 6s linear infinite;
  }
  /* vignette */
  body::after {
    content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 99;
    box-shadow: inset 0 0 180px rgba(0, 0, 0, 0.9);
  }

  /* ── top status bar ──────────────────────────────────────────────────── */
  .topbar {
    position: sticky; top: 0; z-index: 50;
    display: flex; align-items: center; gap: 1.5rem;
    padding: .5rem 1.25rem;
    background: #000;
    border-bottom: 1px solid var(--border-hot);
    font-size: .72rem; letter-spacing: .12em; text-transform: uppercase;
    color: var(--dim);
  }
  .topbar .brand {
    color: var(--phos); font-weight: 700; letter-spacing: .18em;
    text-shadow: 0 0 8px rgba(0, 255, 136, 0.6);
  }
  .topbar .sep { color: var(--mute); }
  .topbar .status-dot {
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    background: var(--phos); margin-right: .35rem;
    box-shadow: 0 0 6px var(--phos);
    animation: pulse-dot 2s infinite;
    vertical-align: middle;
  }
  .topbar .clock { margin-left: auto; color: var(--cyan); letter-spacing: .1em; }

  main {
    max-width: 1040px; margin: 0 auto;
    padding: 1.5rem 1.25rem 4rem;
  }

  /* ── banner ──────────────────────────────────────────────────────────── */
  .banner {
    font-size: .68rem; line-height: 1.1; color: var(--phos);
    text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    white-space: pre; margin: .5rem 0 .25rem;
    user-select: none;
  }
  .subbanner {
    color: var(--dim); font-size: .72rem; letter-spacing: .18em;
    text-transform: uppercase; margin-bottom: 1.5rem;
    padding-bottom: .5rem; border-bottom: 1px dashed var(--mute);
  }
  .subbanner .k { color: var(--cyan); }

  /* ── HUD grid: info + memory + runtime side-by-side ──────────────────── */
  .hud {
    display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem;
    margin-bottom: 1.5rem;
  }
  .panel {
    background: var(--panel);
    border: 1px solid var(--border);
    position: relative;
    padding: .75rem .9rem;
  }
  .panel::before {
    content: attr(data-title);
    position: absolute; top: -.55rem; left: .5rem;
    background: var(--bg); padding: 0 .5rem;
    color: var(--dim); font-size: .64rem; letter-spacing: .2em; text-transform: uppercase;
  }
  .panel::after {
    content: '┐'; position: absolute; top: -.4rem; right: -1px;
    color: var(--border-hot); font-size: 1rem; line-height: 1;
  }
  .kv { display: flex; justify-content: space-between; align-items: baseline; gap: .5rem; }
  .kv .k { color: var(--dim); font-size: .72rem; letter-spacing: .05em; }
  .kv .v { color: var(--fg); font-size: .82rem; font-variant-numeric: tabular-nums; }
  .kv .v.hi { color: var(--phos); text-shadow: 0 0 6px rgba(0, 255, 136, 0.4); }

  /* memory bars */
  .bar-row { display: flex; align-items: center; gap: .5rem; margin: .25rem 0; }
  .bar-row .lbl { color: var(--dim); font-size: .7rem; min-width: 2.5rem; }
  .bar-outer {
    flex: 1; height: 10px; background: rgba(0, 255, 136, 0.06);
    border: 1px solid var(--mute); position: relative;
  }
  .bar-inner {
    height: 100%; background: var(--phos);
    box-shadow: 0 0 6px rgba(0, 255, 136, 0.5);
    transition: width .4s ease;
  }
  .bar-inner.int8 { background: var(--cyan); box-shadow: 0 0 6px rgba(34, 232, 255, 0.5); }
  .bar-inner.int4 { background: var(--amber); box-shadow: 0 0 6px rgba(255, 184, 77, 0.5); }
  .bar-row .size { color: var(--fg); font-size: .72rem; min-width: 5rem; text-align: right; font-variant-numeric: tabular-nums; }
  .bar-row .mult { color: var(--dim); font-size: .68rem; min-width: 2.5rem; text-align: right; }

  /* ── input section ───────────────────────────────────────────────────── */
  .input-row { position: relative; margin-bottom: 1rem; }
  .input-row::before {
    content: '>_'; position: absolute; left: .55rem; top: .55rem;
    color: var(--phos); font-weight: 700; letter-spacing: 0;
    text-shadow: 0 0 6px rgba(0, 255, 136, 0.6); z-index: 1;
  }
  textarea {
    width: 100%; min-height: 72px; resize: vertical;
    background: var(--panel); border: 1px solid var(--border);
    color: var(--fg); padding: .55rem .9rem .55rem 2.3rem;
    font-family: var(--mono); font-size: .9rem;
    caret-color: var(--phos);
  }
  textarea::placeholder { color: var(--mute); }
  textarea:focus { outline: none; border-color: var(--border-hot);
                   box-shadow: 0 0 0 1px var(--border-hot), 0 0 20px rgba(0, 255, 136, 0.1); }

  .controls {
    display: flex; gap: .5rem; align-items: center; flex-wrap: wrap;
    margin-bottom: 1rem;
    padding: .5rem .75rem; background: var(--panel); border: 1px solid var(--border);
  }
  .pill-label { color: var(--dim); font-size: .68rem; letter-spacing: .15em;
                text-transform: uppercase; margin-right: .25rem; }
  input[type=number], select {
    background: #000; border: 1px solid var(--mute); color: var(--phos);
    padding: .25rem .5rem; font: inherit; font-size: .78rem;
    font-family: var(--mono); font-variant-numeric: tabular-nums;
  }
  input[type=number] { width: 60px; }
  input:focus, select:focus { outline: none; border-color: var(--phos); }

  .btn {
    background: transparent; border: 1px solid var(--border-hot);
    color: var(--phos); font: inherit; font-size: .75rem; font-weight: 600;
    padding: .35rem .9rem; letter-spacing: .15em; text-transform: uppercase;
    cursor: pointer; font-family: var(--mono);
    transition: all .15s;
  }
  .btn:hover:not(:disabled) {
    background: rgba(0, 255, 136, 0.1);
    box-shadow: 0 0 12px rgba(0, 255, 136, 0.3);
    color: #fff;
  }
  .btn.primary {
    background: var(--phos); color: #000; border-color: var(--phos);
    box-shadow: 0 0 10px rgba(0, 255, 136, 0.4);
  }
  .btn.primary:hover:not(:disabled) { background: #4fffaa; box-shadow: 0 0 18px rgba(0, 255, 136, 0.7); }
  .btn:disabled { opacity: .3; cursor: not-allowed; }
  .btn.secondary { color: var(--cyan); border-color: var(--cyan); }
  .btn.secondary:hover:not(:disabled) { background: rgba(34, 232, 255, 0.12); color: #fff; box-shadow: 0 0 12px rgba(34, 232, 255, 0.4); }
  .spacer { flex: 1; }
  .hint { color: var(--mute); font-size: .68rem; letter-spacing: .1em; }
  .hint kbd {
    background: #000; border: 1px solid var(--mute); color: var(--cyan);
    padding: 1px 6px; font-size: .66rem; font-family: var(--mono);
  }

  /* ── output ──────────────────────────────────────────────────────────── */
  .output {
    position: relative;
    background: var(--panel); border: 1px solid var(--border);
    padding: 1rem 1.1rem;
    min-height: 180px; white-space: pre-wrap; word-wrap: break-word;
    font-size: .88rem; line-height: 1.7;
  }
  .output::before {
    content: 'OUTPUT'; position: absolute; top: -.55rem; left: .5rem;
    background: var(--bg); padding: 0 .5rem;
    color: var(--dim); font-size: .64rem; letter-spacing: .2em;
  }
  .output .prompt { color: var(--dim); }
  .output .gen { color: var(--fg); text-shadow: 0 0 4px rgba(199, 245, 219, 0.15); }
  .tok-flash { animation: token-flash .5s ease-out; }
  .caret {
    display: inline-block; width: 8px; height: 1em; background: var(--phos);
    vertical-align: text-bottom; animation: caret-blink .8s infinite;
    box-shadow: 0 0 6px var(--phos);
  }

  /* ── live stats hud ──────────────────────────────────────────────────── */
  .livestats {
    margin-top: .75rem;
    display: grid; grid-template-columns: auto 1fr auto auto auto; gap: 1rem;
    align-items: center;
    padding: .5rem .9rem;
    background: #000; border: 1px solid var(--border);
    font-size: .72rem; letter-spacing: .08em;
  }
  .livestats .stat-k { color: var(--dim); text-transform: uppercase; }
  .livestats .stat-v {
    color: var(--phos); font-variant-numeric: tabular-nums;
    text-shadow: 0 0 4px rgba(0, 255, 136, 0.4);
  }
  .livestats .stat-v.cyan { color: var(--cyan); text-shadow: 0 0 4px rgba(34, 232, 255, 0.4); }
  .livestats .stat-v.amber { color: var(--amber); text-shadow: 0 0 4px rgba(255, 184, 77, 0.4); }
  .livestats .err { color: var(--red); }
  .spark {
    display: flex; align-items: flex-end; height: 20px; gap: 1px;
  }
  .spark span {
    display: inline-block; width: 3px; background: var(--phos);
    box-shadow: 0 0 3px var(--phos);
  }

  /* ── compare mode ────────────────────────────────────────────────────── */
  .compare { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-top: 1rem; }
  .compare .col {
    background: var(--panel); border: 1px solid var(--border);
    padding: .9rem; display: flex; flex-direction: column; gap: .55rem;
    position: relative;
  }
  .compare .col::before {
    content: attr(data-q); position: absolute; top: -.55rem; left: .5rem;
    background: var(--bg); padding: 0 .5rem;
    font-size: .68rem; letter-spacing: .18em; text-transform: uppercase;
  }
  .compare .col.fp32::before { color: var(--phos); }
  .compare .col.int8::before { color: var(--cyan); }
  .compare .col.int4::before { color: var(--amber); }
  .compare .col.fp32 { border-color: var(--border-hot); }
  .compare .col.int8 { border-color: #1f4958; }
  .compare .col.int4 { border-color: #5c4420; }
  .compare .perf {
    display: flex; justify-content: space-between; font-size: .68rem;
    color: var(--dim); letter-spacing: .1em;
    padding-bottom: .5rem; border-bottom: 1px dashed var(--mute);
  }
  .compare .perf .v { color: var(--fg); font-variant-numeric: tabular-nums; }
  .compare .text {
    font-size: .78rem; line-height: 1.65; white-space: pre-wrap; word-wrap: break-word;
    min-height: 120px; max-height: 300px; overflow-y: auto; color: var(--fg);
  }
  .compare .text .prompt { color: var(--dim); }

  /* ── boot overlay ────────────────────────────────────────────────────── */
  .boot {
    position: fixed; inset: 0; background: var(--bg); z-index: 200;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono);
  }
  .boot .lines { max-width: 540px; width: 100%; padding: 0 2rem; }
  .boot .line { color: var(--phos); font-size: .8rem; margin: .3rem 0;
                opacity: 0; overflow: hidden; white-space: nowrap; }
  .boot .line.ok::after { content: ' [OK]'; color: var(--cyan); }
  .boot.hide { opacity: 0; pointer-events: none; transition: opacity .3s; }

  @media (max-width: 820px) {
    .hud { grid-template-columns: 1fr; }
    .compare { grid-template-columns: 1fr; }
    .livestats { grid-template-columns: 1fr 1fr; }
  }
</style>
</head>
<body>

<!-- boot overlay plays once on load -->
<div class="boot" id="boot">
  <div class="lines" id="bootLines"></div>
</div>

<!-- top status bar -->
<div class="topbar">
  <span><span class="status-dot"></span><span class="brand">MINILLM.CPP</span></span>
  <span class="sep">│</span>
  <span>ARM NEON</span>
  <span class="sep">│</span>
  <span>SMOLLM2-135M</span>
  <span class="sep">│</span>
  <span>LOOPBACK :8080</span>
  <span class="clock" id="clock">--:--:--</span>
</div>

<main>
  <pre class="banner" id="banner">
   __  __ ___ _  _ ___ _    _    __  __    ___ ___ ___
  |  \/  |_ _| \| |_ _| |  | |  |  \/  |  / __| _ \ _ \
  | |\/| || || .` || || |__| |__| |\/| | | (__|  _/  _/
  |_|  |_|___|_|\_|___|____|____|_|  |_|  \___|_| |_|   v1.0.0</pre>
  <div class="subbanner">
    <span class="k">●</span> CPU-ONLY · PER-CHANNEL QUANTIZATION · ZERO EXTERNAL DEPS · 97 TESTS PASS
  </div>

  <!-- HUD: three info panels side-by-side -->
  <div class="hud">
    <div class="panel" data-title="arch">
      <div class="kv"><span class="k">vocab</span><span class="v" id="iVocab">—</span></div>
      <div class="kv"><span class="k">layers</span><span class="v" id="iLayers">—</span></div>
      <div class="kv"><span class="k">hidden</span><span class="v" id="iHidden">—</span></div>
      <div class="kv"><span class="k">heads</span><span class="v" id="iHeads">—</span></div>
      <div class="kv"><span class="k">kv-heads</span><span class="v" id="iKvHeads">—</span></div>
      <div class="kv"><span class="k">head-dim</span><span class="v" id="iHeadDim">—</span></div>
    </div>
    <div class="panel" data-title="memory">
      <div class="bar-row"><span class="lbl">fp32</span>
        <div class="bar-outer"><div class="bar-inner" id="barFp32" style="width:100%"></div></div>
        <span class="size" id="sizeFp32">—</span><span class="mult">1×</span></div>
      <div class="bar-row"><span class="lbl">int8</span>
        <div class="bar-outer"><div class="bar-inner int8" id="barInt8" style="width:0"></div></div>
        <span class="size" id="sizeInt8">—</span><span class="mult" id="multInt8">—</span></div>
      <div class="bar-row"><span class="lbl">int4</span>
        <div class="bar-outer"><div class="bar-inner int4" id="barInt4" style="width:0"></div></div>
        <span class="size" id="sizeInt4">—</span><span class="mult" id="multInt4">—</span></div>
    </div>
    <div class="panel" data-title="runtime">
      <div class="kv"><span class="k">status</span><span class="v hi">READY</span></div>
      <div class="kv"><span class="k">quant</span><span class="v">per-channel symm.</span></div>
      <div class="kv"><span class="k">simd</span><span class="v">NEON ×4 FMA</span></div>
      <div class="kv"><span class="k">kv-cache</span><span class="v">per-layer append</span></div>
      <div class="kv"><span class="k">sampling</span><span class="v">greedy argmax</span></div>
      <div class="kv"><span class="k">stream</span><span class="v">SSE token</span></div>
    </div>
  </div>

  <div class="input-row">
    <textarea id="prompt" placeholder="enter prompt and press ⌘⏎">The capital of France is</textarea>
  </div>

  <div class="controls">
    <span class="pill-label">tokens</span>
    <input id="maxTokens" type="number" value="48" min="1" max="256">
    <span class="pill-label" style="margin-left:.5rem">quant</span>
    <select id="quant">
      <option value="fp32" selected>fp32</option>
      <option value="int8">int8</option>
      <option value="int4">int4</option>
    </select>
    <span class="hint" style="margin-left:1rem"><kbd>⌘⏎</kbd> run · <kbd>esc</kbd> stop</span>
    <div class="spacer"></div>
    <button class="btn secondary" id="compareBtn">▣ COMPARE</button>
    <button class="btn primary"   id="goBtn">▶ EXECUTE</button>
  </div>

  <div class="output" id="out"><span class="prompt">// awaiting input</span></div>

  <div class="livestats">
    <span><span class="stat-k">TOK/S</span></span>
    <span class="spark" id="spark"></span>
    <span><span class="stat-k">GEN </span><span class="stat-v" id="statN">0</span></span>
    <span><span class="stat-k">ELAPSED </span><span class="stat-v cyan" id="statMs">0ms</span></span>
    <span><span class="stat-k">QUANT </span><span class="stat-v amber" id="statQ">—</span></span>
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

let activeStream = null;
let tpsHistory = [];

// ── boot sequence ───────────────────────────────────────────────────────────
async function playBoot() {
  const lines = [
    '> initializing minillm runtime',
    '> loading model weights into memory',
    '> building rope cache [max_pos=2048, head_dim=64]',
    '> opening tokenizer: vocab=49152',
    '> starting http server on loopback:8080',
    '> system ready',
  ];
  const container = $('bootLines');
  for (const text of lines) {
    const div = document.createElement('div');
    div.className = 'line';
    div.textContent = text;
    container.appendChild(div);
    div.style.opacity = 1;
    div.style.width = '0';
    const target = text.length + 'ch';
    div.style.transition = 'width .18s steps(' + text.length + ')';
    requestAnimationFrame(() => { div.style.width = target; });
    await sleep(170);
    div.classList.add('ok');
  }
  await sleep(300);
  $('boot').classList.add('hide');
  setTimeout(() => $('boot').remove(), 400);
}
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// ── clock ───────────────────────────────────────────────────────────────────
function tickClock() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  $('clock').textContent = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}
setInterval(tickClock, 1000); tickClock();

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
function fmtMB(b) { return (b/1024/1024).toFixed(1) + 'MB'; }

// ── sparkline ───────────────────────────────────────────────────────────────
function pushSpark(tps) {
  tpsHistory.push(tps);
  if (tpsHistory.length > 40) tpsHistory.shift();
  const max = Math.max(1, ...tpsHistory);
  spark.innerHTML = '';
  for (const v of tpsHistory) {
    const s = document.createElement('span');
    s.style.height = Math.max(2, 20 * v / max) + 'px';
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
  $('statQ').textContent  = quant;

  outEl.innerHTML = '<span class="prompt"></span><span class="gen"></span><span class="caret"></span>';
  outEl.querySelector('.prompt').textContent = prompt;
  const genSpan = outEl.querySelector('.gen');

  const url = `/stream?prompt=${encodeURIComponent(prompt)}&max_tokens=${max_tokens}&quant=${quant}`;
  const es  = new EventSource(url);
  activeStream = es;

  const t0 = performance.now();

  es.addEventListener('token', (e) => {
    const d = JSON.parse(e.data);
    // Append the delta as a flashing span, then collapse once animation ends
    const flash = document.createElement('span');
    flash.className = 'tok-flash';
    flash.textContent = d.delta;
    genSpan.appendChild(flash);
    setTimeout(() => {
      // merge the flashed span's text into the parent to keep DOM small
      const t = flash.textContent;
      flash.replaceWith(document.createTextNode(t));
    }, 500);

    const elapsed = (performance.now() - t0) / 1000;
    const tps = d.n / elapsed;
    pushSpark(tps);
    $('statN').textContent  = d.n;
    $('statMs').textContent = Math.round(elapsed * 1000) + 'ms';
  });

  es.addEventListener('done', (e) => {
    const d = JSON.parse(e.data);
    outEl.querySelector('.caret')?.remove();
    $('statN').textContent  = d.n_tokens;
    $('statMs').textContent = d.ms + 'ms';
    es.close(); activeStream = null; goBtn.disabled = false;
  });

  es.onerror = () => {
    outEl.querySelector('.caret')?.remove();
    $('statMs').innerHTML = '<span class="err">LOST</span>';
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
  outEl.innerHTML = '<span class="prompt">// running three quantization levels in parallel…</span>';
  compareEl.innerHTML = '';
  compareEl.style.display = 'grid';
  $('statQ').textContent = 'all';

  try {
    const r = await fetch('/compare', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ prompt, max_tokens }),
    });
    const data = await r.json();
    compareEl.innerHTML = '';
    for (const q of ['fp32', 'int8', 'int4']) {
      const d = data[q];
      const col = document.createElement('div');
      col.className = 'col ' + q;
      col.dataset.q = q;
      col.innerHTML = `
        <div class="perf">
          <span>${d.n_tokens} tok · <span class="v">${d.ms}ms</span></span>
          <span><span class="v">${d.tok_per_sec}</span> tok/s</span>
        </div>
        <div class="text"><span class="prompt">${escapeHtml(prompt)}</span>${escapeHtml(d.generated)}</div>
      `;
      compareEl.appendChild(col);
    }
    outEl.innerHTML = '<span class="prompt">// comparison complete — scroll for 3-way diff</span>';
  } catch (e) {
    outEl.innerHTML = `<span style="color: var(--red)">// error: ${escapeHtml(e.message)}</span>`;
  } finally {
    compareBtn.disabled = false; goBtn.disabled = false;
  }
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

// init
loadInfo();
playBoot();
</script>
</body>
</html>
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
                               const std::string& prompt, int max_tokens) {
    auto prompt_ids = encode(st.tok, prompt);
    auto t0 = clk::now();
    auto ids = generate(w, st.cfg, st.rope, prompt_ids, max_tokens, st.tok.eos_id);
    int ms = static_cast<int>(std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    double tps = ids.empty() ? 0 : static_cast<double>(ids.size()) / (ms / 1000.0);
    return { decode(st.tok, ids), static_cast<int>(ids.size()), ms, static_cast<int>(tps) };
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
    return http_response(200, "application/json", j.dump());
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
    auto r = run_generate(st, w, prompt, max_tokens);
    json j = {{"prompt",prompt},{"generated",r.generated},{"n_tokens",r.n_tokens},
              {"ms",r.ms},{"tok_per_sec",r.tok_per_sec}};
    return http_response(200, "application/json", j.dump());
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

    json result;
    for (const char* q : {"fp32", "int8", "int4"}) {
        const ModelWeights& w = ensure_weights(st, q);
        auto r = run_generate(st, w, prompt, max_tokens);
        result[q] = {{"generated",r.generated},{"n_tokens",r.n_tokens},
                     {"ms",r.ms},{"tok_per_sec",r.tok_per_sec}};
    }
    return http_response(200, "application/json", result.dump());
}

// SSE: inline the generate loop so we can push one token at a time.
static void handle_stream(ServerState& st, int sock, const std::string& qs) {
    std::string prompt     = query_param(qs, "prompt");
    int         max_tokens = std::max(1, std::min(256, std::atoi(query_param(qs, "max_tokens").c_str())));
    std::string quant      = query_param(qs, "quant");
    if (quant.empty()) quant = "fp32";

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

    for (int step = 0; step < max_tokens; ++step) {
        const float* lp = logits.data_as<float>();
        int32_t next_id = 0; float best = lp[0];
        for (size_t v = 1; v < vocab; ++v)
            if (lp[v] > best) { best = lp[v]; next_id = static_cast<int32_t>(v); }

        gen_ids.push_back(next_id);

        // Decode cumulative then take the new suffix.
        std::string full = decode(st.tok, gen_ids);
        std::string delta = full.substr(prev_text.size());
        prev_text = full;

        json ev = {{"delta",delta},{"id",next_id},{"n",gen_ids.size()}};
        emit("token", ev.dump());

        if (next_id == st.tok.eos_id) break;
        logits = forward_one(w, st.cfg, st.rope, next_id, kv);
    }

    int ms = static_cast<int>(std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    double tps = gen_ids.empty() ? 0 : static_cast<double>(gen_ids.size()) / (ms / 1000.0);
    json done = {{"n_tokens",gen_ids.size()},{"ms",ms},{"tok_per_sec",static_cast<int>(tps)}};
    emit("done", done.dump());
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
