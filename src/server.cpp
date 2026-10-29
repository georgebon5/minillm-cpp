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
<title>minillm.cpp</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #0a0a0a;
    --panel: #111;
    --border: #1f1f1f;
    --fg: #e8e8e8;
    --muted: #6b7280;
    --accent: #10b981;
    --mono: ui-monospace, "SF Mono", Menlo, monospace;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--fg); }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Inter", system-ui, sans-serif;
    font-size: 14px; line-height: 1.5;
    min-height: 100vh;
    display: flex; justify-content: center;
  }
  main { width: 100%; max-width: 820px; padding: 3rem 2rem 4rem; }

  header { display: flex; justify-content: space-between; align-items: baseline;
           margin-bottom: 2rem; }
  h1 { margin: 0; font-size: 1.1rem; font-weight: 500; letter-spacing: -.01em; }
  h1 .mono { font-family: var(--mono); color: var(--accent); }
  .tagline { color: var(--muted); font-size: .8rem; }

  .info-toggle {
    background: none; border: none; color: var(--muted); cursor: pointer;
    font: inherit; padding: 0; letter-spacing: .02em;
  }
  .info-toggle:hover { color: var(--fg); }
  .info-panel {
    display: none; margin-bottom: 1.5rem; padding: 1rem 1.25rem;
    background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
    font-family: var(--mono); font-size: .78rem; line-height: 1.7;
  }
  .info-panel.open { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.25rem; }
  .info-panel .label { color: var(--muted); }
  .info-panel .val { color: var(--fg); }
  .info-panel h3 { margin: 0 0 .35rem 0; font-size: .72rem; font-weight: 500;
                   color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }

  textarea {
    width: 100%; min-height: 72px; resize: vertical;
    background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
    color: var(--fg); padding: .75rem .9rem;
    font-family: var(--mono); font-size: .9rem;
  }
  textarea:focus { outline: none; border-color: #333; }

  .controls {
    display: flex; gap: .75rem; align-items: center; flex-wrap: wrap;
    margin: 1rem 0 .5rem;
  }
  .control { display: flex; align-items: center; gap: .4rem; color: var(--muted);
             font-size: .82rem; }
  input[type=number], select {
    background: var(--panel); border: 1px solid var(--border); border-radius: 4px;
    color: var(--fg); padding: .25rem .5rem; font: inherit; font-size: .82rem;
  }
  input[type=number] { width: 60px; }
  input:focus, select:focus { outline: none; border-color: #333; }

  button {
    background: var(--fg); border: none; color: var(--bg);
    font: inherit; font-weight: 500; font-size: .85rem;
    padding: .45rem 1rem; border-radius: 4px; cursor: pointer;
  }
  button:disabled { opacity: .4; cursor: not-allowed; }
  button:hover:not(:disabled) { background: #fff; }
  button.ghost { background: transparent; color: var(--fg); border: 1px solid var(--border); }
  button.ghost:hover:not(:disabled) { background: var(--panel); }
  .spacer { flex: 1; }
  kbd {
    font-family: var(--mono); font-size: .72rem; background: var(--panel);
    padding: 1px 6px; border: 1px solid var(--border); border-radius: 3px;
    color: var(--muted);
  }

  .output {
    margin-top: 1rem; padding: 1rem 1.1rem;
    background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
    min-height: 180px; white-space: pre-wrap; word-wrap: break-word;
    font-family: var(--mono); font-size: .88rem; line-height: 1.6;
  }
  .output .prompt { color: var(--muted); }
  .output .gen { color: var(--fg); }
  .caret { display: inline-block; width: 7px; height: 1em; background: var(--accent);
           vertical-align: text-bottom; animation: blink 1s steps(2) infinite; }
  @keyframes blink { 50% { opacity: 0; } }

  .status {
    margin-top: .75rem; color: var(--muted); font-size: .78rem;
    font-family: var(--mono); font-variant-numeric: tabular-nums;
    display: flex; gap: 1.5rem; flex-wrap: wrap;
  }
  .status .val { color: var(--fg); }
  .status .err { color: #ef4444; }

  /* compare mode */
  .compare { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-top: 1rem; }
  .compare .col {
    background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
    padding: .85rem; display: flex; flex-direction: column; gap: .5rem;
  }
  .compare h4 { margin: 0; font-size: .72rem; font-weight: 500;
                color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }
  .compare h4 .pill {
    background: var(--border); color: var(--fg); padding: 2px 8px;
    border-radius: 3px; font-family: var(--mono); margin-left: .35rem;
  }
  .compare .text {
    font-family: var(--mono); font-size: .78rem; line-height: 1.55;
    white-space: pre-wrap; word-wrap: break-word;
    min-height: 120px; max-height: 320px; overflow-y: auto;
    color: var(--fg);
  }
  .compare .text .prompt { color: var(--muted); }
  .compare .foot {
    font-family: var(--mono); font-size: .72rem; color: var(--muted);
    display: flex; justify-content: space-between;
    border-top: 1px solid var(--border); padding-top: .5rem;
  }
  .compare .foot .val { color: var(--fg); }

  @media (max-width: 640px) {
    main { padding: 1.5rem 1rem 3rem; }
    .info-panel.open { grid-template-columns: 1fr; }
    .compare { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>minillm<span class="mono">.cpp</span></h1>
      <div class="tagline">C++20 · SmolLM2-135M · ARM NEON · quantization</div>
    </div>
    <button class="info-toggle" id="infoToggle">model info ▾</button>
  </header>

  <div class="info-panel" id="infoPanel">
    <div>
      <h3>Architecture</h3>
      <div><span class="label">vocab   </span><span class="val" id="iVocab">—</span></div>
      <div><span class="label">layers  </span><span class="val" id="iLayers">—</span></div>
      <div><span class="label">hidden  </span><span class="val" id="iHidden">—</span></div>
      <div><span class="label">heads   </span><span class="val" id="iHeads">—</span></div>
      <div><span class="label">kv-heads</span><span class="val" id="iKvHeads">—</span></div>
      <div><span class="label">head-dim</span><span class="val" id="iHeadDim">—</span></div>
    </div>
    <div>
      <h3>Weights memory</h3>
      <div><span class="label">fp32    </span><span class="val" id="mFp32">—</span></div>
      <div><span class="label">int8    </span><span class="val" id="mInt8">—</span></div>
      <div><span class="label">int4    </span><span class="val" id="mInt4">—</span></div>
    </div>
    <div>
      <h3>Runtime</h3>
      <div><span class="label">status  </span><span class="val" style="color: var(--accent)">ready</span></div>
      <div><span class="label">quant   </span><span class="val">per-channel symmetric</span></div>
      <div><span class="label">simd    </span><span class="val">ARM NEON (4× FMA acc)</span></div>
    </div>
  </div>

  <textarea id="prompt" placeholder="prompt…">The capital of France is</textarea>

  <div class="controls">
    <span class="control">tokens
      <input id="maxTokens" type="number" value="48" min="1" max="256">
    </span>
    <span class="control">quant
      <select id="quant">
        <option value="fp32" selected>fp32</option>
        <option value="int8">int8</option>
        <option value="int4">int4</option>
      </select>
    </span>
    <div class="spacer"></div>
    <button class="ghost" id="compareBtn">compare all</button>
    <button id="goBtn">generate</button>
  </div>
  <div class="status"><kbd>⌘</kbd><kbd>⏎</kbd> submit · <kbd>esc</kbd> stop</div>

  <div class="output" id="out"><span class="prompt">(output appears here)</span></div>
  <div class="status" id="status"></div>

  <div class="compare" id="compare" style="display:none"></div>
</main>

<script>
const $ = (id) => document.getElementById(id);
const promptEl = $('prompt');
const outEl    = $('out');
const statusEl = $('status');
const compareEl = $('compare');
const goBtn   = $('goBtn');
const compareBtn = $('compareBtn');

let activeStream = null;

// ── fetch and render model info on load ─────────────────────────────────────
async function loadInfo() {
  try {
    const r = await fetch('/info');
    const d = await r.json();
    $('iVocab').textContent   = d.vocab.toLocaleString();
    $('iLayers').textContent  = d.layers;
    $('iHidden').textContent  = d.hidden;
    $('iHeads').textContent   = d.n_heads;
    $('iKvHeads').textContent = d.n_kv_heads;
    $('iHeadDim').textContent = d.head_dim;
    $('mFp32').textContent = fmtMB(d.mem_fp32_bytes);
    $('mInt8').textContent = fmtMB(d.mem_int8_bytes) + '  (' + (d.mem_fp32_bytes/d.mem_int8_bytes).toFixed(1) + '×)';
    $('mInt4').textContent = fmtMB(d.mem_int4_bytes) + '  (' + (d.mem_fp32_bytes/d.mem_int4_bytes).toFixed(1) + '×)';
  } catch (e) { console.warn('info load failed', e); }
}
function fmtMB(b) { return (b / 1024 / 1024).toFixed(1) + ' MB'; }

$('infoToggle').addEventListener('click', () => {
  $('infoPanel').classList.toggle('open');
  const open = $('infoPanel').classList.contains('open');
  $('infoToggle').textContent = open ? 'model info ▴' : 'model info ▾';
});

// ── streaming generation (SSE) ──────────────────────────────────────────────
function streamGenerate() {
  stopStream();
  compareEl.style.display = 'none';

  const prompt     = promptEl.value;
  const max_tokens = parseInt($('maxTokens').value, 10);
  const quant      = $('quant').value;

  if (!prompt.trim()) return;

  goBtn.disabled = true;
  statusEl.innerHTML = '<span>streaming…</span>';
  outEl.innerHTML = '<span class="prompt"></span><span class="gen"></span><span class="caret"></span>';
  outEl.querySelector('.prompt').textContent = prompt;
  const genSpan = outEl.querySelector('.gen');

  const url = `/stream?prompt=${encodeURIComponent(prompt)}&max_tokens=${max_tokens}&quant=${quant}`;
  const es = new EventSource(url);
  activeStream = es;

  let tokCount = 0;
  const t0 = performance.now();

  es.addEventListener('token', (e) => {
    const d = JSON.parse(e.data);
    genSpan.textContent += d.delta;
    tokCount = d.n;
    const elapsed = (performance.now() - t0) / 1000;
    statusEl.innerHTML =
      `<span>streaming </span><span class="val">${tokCount}</span>` +
      `<span> · </span><span class="val">${(tokCount/elapsed).toFixed(0)}</span><span> tok/s</span>` +
      `<span> · </span><span class="val">${quant}</span>`;
  });

  es.addEventListener('done', (e) => {
    const d = JSON.parse(e.data);
    outEl.querySelector('.caret')?.remove();
    statusEl.innerHTML =
      `<span>done </span><span class="val">${d.n_tokens}</span><span> tokens · </span>` +
      `<span class="val">${d.ms}</span><span> ms · </span>` +
      `<span class="val">${d.tok_per_sec}</span><span> tok/s · </span>` +
      `<span class="val">${quant}</span>`;
    es.close();
    activeStream = null;
    goBtn.disabled = false;
  });

  es.onerror = () => {
    outEl.querySelector('.caret')?.remove();
    statusEl.innerHTML = '<span class="err">connection lost</span>';
    es.close();
    activeStream = null;
    goBtn.disabled = false;
  };
}

function stopStream() {
  if (activeStream) { activeStream.close(); activeStream = null; }
  outEl.querySelector('.caret')?.remove();
  goBtn.disabled = false;
}

// ── compare all three quant levels ──────────────────────────────────────────
async function compareAll() {
  stopStream();
  const prompt     = promptEl.value;
  const max_tokens = parseInt($('maxTokens').value, 10);
  if (!prompt.trim()) return;

  compareBtn.disabled = true;
  goBtn.disabled = true;
  statusEl.innerHTML = '<span>running all three (fp32, int8, int4)…</span>';
  outEl.innerHTML = '<span class="prompt">(see comparison below)</span>';
  compareEl.innerHTML = '';
  compareEl.style.display = 'grid';

  try {
    const r = await fetch('/compare', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ prompt, max_tokens }),
    });
    const data = await r.json();
    compareEl.innerHTML = '';
    for (const q of ['fp32', 'int8', 'int4']) {
      const d = data[q];
      const col = document.createElement('div');
      col.className = 'col';
      col.innerHTML = `
        <h4>${q}<span class="pill">${d.tok_per_sec} tok/s</span></h4>
        <div class="text"><span class="prompt">${escapeHtml(prompt)}</span>${escapeHtml(d.generated)}</div>
        <div class="foot"><span>${d.n_tokens} tokens</span><span class="val">${d.ms} ms</span></div>
      `;
      compareEl.appendChild(col);
    }
    statusEl.innerHTML = '<span>comparison complete</span>';
  } catch (e) {
    statusEl.innerHTML = `<span class="err">error: ${escapeHtml(e.message)}</span>`;
  } finally {
    compareBtn.disabled = false;
    goBtn.disabled = false;
  }
}

function escapeHtml(s) {
  return (s || '').replace(/[&<>"']/g, c => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  }[c]));
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
