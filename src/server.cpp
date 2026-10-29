// Minimal HTTP server exposing the inference engine over localhost.
// Serves a single HTML page at GET / and handles POST /generate.
// No external HTTP library — raw POSIX sockets, single-threaded, one request at a time.

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

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace minillm;
namespace fs = std::filesystem;
using json   = nlohmann::json;
using clk    = std::chrono::steady_clock;

// ── Embedded HTML frontend (single page) ─────────────────────────────────────
static const char* kIndexHtml = R"HTML(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>minillm.cpp</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 2rem;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    background: #0b0d10; color: #e6e9ef;
    display: flex; justify-content: center;
  }
  main { width: 100%; max-width: 720px; }
  h1 { margin: 0 0 .25rem 0; font-weight: 600; letter-spacing: -.02em; }
  .sub { color: #8a93a0; font-size: .9rem; margin-bottom: 1.5rem; }
  textarea, select, input, button {
    font: inherit; color: inherit; background: #151a21;
    border: 1px solid #262c36; border-radius: 8px; padding: .65rem .8rem;
  }
  textarea {
    width: 100%; min-height: 90px; resize: vertical;
    font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: .95rem;
  }
  textarea:focus, select:focus, input:focus {
    outline: none; border-color: #4a90e2;
  }
  .controls {
    display: flex; gap: .75rem; align-items: center; margin: .75rem 0 1rem 0;
    flex-wrap: wrap;
  }
  label { font-size: .85rem; color: #8a93a0; }
  input[type=number] { width: 90px; }
  button {
    background: #4a90e2; border-color: #4a90e2; color: white; cursor: pointer;
    font-weight: 500; padding: .55rem 1.2rem;
  }
  button:disabled { opacity: .5; cursor: not-allowed; }
  button:hover:not(:disabled) { background: #5aa0f2; }
  .output {
    background: #151a21; border: 1px solid #262c36; border-radius: 8px;
    padding: 1rem; min-height: 180px; white-space: pre-wrap;
    font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: .95rem;
    line-height: 1.5;
  }
  .output .prompt { color: #8a93a0; }
  .output .gen    { color: #e6e9ef; }
  .status { color: #8a93a0; font-size: .85rem; margin-top: .75rem; font-variant-numeric: tabular-nums; }
  .error { color: #ff6b6b; }
</style>
</head>
<body>
<main>
  <h1>minillm.cpp</h1>
  <div class="sub">
    Hand-written C++ LLM inference engine · SmolLM2-135M · ARM NEON
  </div>

  <textarea id="prompt" placeholder="Enter a prompt…">The capital of France is</textarea>

  <div class="controls">
    <label>max tokens
      <input id="max_tokens" type="number" value="48" min="1" max="512">
    </label>
    <label>quant
      <select id="quant">
        <option value="fp32" selected>fp32</option>
        <option value="int8">int8</option>
        <option value="int4">int4</option>
      </select>
    </label>
    <button id="go">Generate</button>
  </div>

  <div class="output" id="out"><span class="prompt">(output will appear here)</span></div>
  <div class="status" id="status"></div>
</main>

<script>
const btn    = document.getElementById('go');
const out    = document.getElementById('out');
const status = document.getElementById('status');

async function generate() {
  const prompt     = document.getElementById('prompt').value;
  const max_tokens = parseInt(document.getElementById('max_tokens').value, 10);
  const quant      = document.getElementById('quant').value;

  btn.disabled = true;
  status.textContent = 'generating…';
  out.innerHTML = `<span class="prompt">${escapeHtml(prompt)}</span><span class="gen">▍</span>`;

  try {
    const t0 = performance.now();
    const resp = await fetch('/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, max_tokens, quant }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    const elapsed = (performance.now() - t0).toFixed(0);
    out.innerHTML =
      `<span class="prompt">${escapeHtml(data.prompt)}</span>` +
      `<span class="gen">${escapeHtml(data.generated)}</span>`;
    status.textContent =
      `${data.n_tokens} tokens · ${data.ms} ms (server) · ${elapsed} ms (round-trip) · ` +
      `${data.tok_per_sec} tok/s · ${quant}`;
  } catch (e) {
    status.innerHTML = `<span class="error">error: ${escapeHtml(e.message)}</span>`;
    out.innerHTML = '<span class="prompt">(failed)</span>';
  } finally {
    btn.disabled = false;
  }
}

function escapeHtml(s) {
  return (s || '').replace(/[&<>"']/g, c => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  }[c]));
}

btn.addEventListener('click', generate);
document.getElementById('prompt').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) generate();
});
</script>
</body>
</html>
)HTML";

// ── Globals set once at startup ──────────────────────────────────────────────
struct ServerState {
    ModelConfig  cfg;
    ModelWeights w_fp32;
    ModelWeights w_int8;   // lazily built on first int8 request
    ModelWeights w_int4;   // lazily built on first int4 request
    bool         int8_ready = false;
    bool         int4_ready = false;
    RopeCache    rope;
    Tokenizer    tok;
};

// ── HTTP helpers (minimal) ────────────────────────────────────────────────────

static std::string http_response(int status, const std::string& ctype,
                                  const std::string& body) {
    const char* status_text = (status == 200) ? "OK"
                            : (status == 400) ? "Bad Request"
                            : (status == 404) ? "Not Found"
                                              : "Internal Server Error";
    std::string resp;
    resp += "HTTP/1.1 " + std::to_string(status) + " " + status_text + "\r\n";
    resp += "Content-Type: " + ctype + "\r\n";
    resp += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    resp += "Connection: close\r\n\r\n";
    resp += body;
    return resp;
}

// Read a full HTTP request from the socket into a string. Returns empty on error.
// Reads headers, parses Content-Length, then reads body.
static std::string read_request(int sock) {
    std::string buf;
    char        chunk[4096];
    size_t      header_end = std::string::npos;

    // Read until we see "\r\n\r\n"
    while (header_end == std::string::npos) {
        ssize_t n = recv(sock, chunk, sizeof(chunk), 0);
        if (n <= 0) return {};
        buf.append(chunk, static_cast<size_t>(n));
        header_end = buf.find("\r\n\r\n");
    }

    // Parse Content-Length if present
    size_t content_len = 0;
    {
        const std::string key = "Content-Length:";
        size_t pos = buf.find(key);
        if (pos != std::string::npos && pos < header_end) {
            size_t eol = buf.find("\r\n", pos);
            std::string val = buf.substr(pos + key.size(), eol - pos - key.size());
            content_len = static_cast<size_t>(std::stoul(val));
        }
    }

    // Read remaining body
    size_t body_start = header_end + 4;
    while (buf.size() - body_start < content_len) {
        ssize_t n = recv(sock, chunk, sizeof(chunk), 0);
        if (n <= 0) break;
        buf.append(chunk, static_cast<size_t>(n));
    }
    return buf;
}

// ── Request handling ─────────────────────────────────────────────────────────

static std::string handle_generate(ServerState& st, const std::string& body_json) {
    json req;
    try { req = json::parse(body_json); }
    catch (const std::exception& e) {
        return http_response(400, "application/json",
                             json{{"error", std::string("bad json: ") + e.what()}}.dump());
    }

    std::string prompt     = req.value("prompt", "");
    int         max_tokens = req.value("max_tokens", 48);
    std::string quant      = req.value("quant", "fp32");
    if (prompt.empty())
        return http_response(400, "application/json",
                             json{{"error", "empty prompt"}}.dump());
    max_tokens = std::max(1, std::min(512, max_tokens));

    // Choose weights (lazy quantize)
    const ModelWeights* w = &st.w_fp32;
    if (quant == "int8") {
        if (!st.int8_ready) {
            std::cerr << "Quantizing to INT8 (one-time)...\n";
            st.w_int8     = quantize_weights(st.w_fp32, DType::I8);
            st.int8_ready = true;
        }
        w = &st.w_int8;
    } else if (quant == "int4") {
        if (!st.int4_ready) {
            std::cerr << "Quantizing to INT4 (one-time)...\n";
            st.w_int4     = quantize_weights(st.w_fp32, DType::I4);
            st.int4_ready = true;
        }
        w = &st.w_int4;
    }

    auto prompt_ids = encode(st.tok, prompt);

    auto t0 = clk::now();
    auto gen_ids = generate(*w, st.cfg, st.rope, prompt_ids, max_tokens, st.tok.eos_id);
    double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();

    std::string generated = decode(st.tok, gen_ids);
    double tps = static_cast<double>(gen_ids.size()) / (ms / 1000.0);

    json resp = {
        {"prompt",      prompt},
        {"generated",   generated},
        {"n_tokens",    gen_ids.size()},
        {"ms",          static_cast<int>(ms)},
        {"tok_per_sec", static_cast<int>(tps)},
    };
    return http_response(200, "application/json", resp.dump());
}

static std::string handle_request(ServerState& st, const std::string& raw) {
    // Parse request line
    size_t eol = raw.find("\r\n");
    if (eol == std::string::npos) return http_response(400, "text/plain", "bad request");

    std::string line = raw.substr(0, eol);
    size_t sp1 = line.find(' ');
    size_t sp2 = line.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos)
        return http_response(400, "text/plain", "bad request line");

    std::string method = line.substr(0, sp1);
    std::string path   = line.substr(sp1 + 1, sp2 - sp1 - 1);

    if (method == "GET" && (path == "/" || path == "/index.html"))
        return http_response(200, "text/html; charset=utf-8", kIndexHtml);

    if (method == "POST" && path == "/generate") {
        size_t header_end = raw.find("\r\n\r\n");
        std::string body  = (header_end != std::string::npos)
                                ? raw.substr(header_end + 4) : "";
        return handle_generate(st, body);
    }

    return http_response(404, "text/plain", "not found");
}

// ── main ─────────────────────────────────────────────────────────────────────

static void usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " --model <dir> [--port 8080]\n";
}

int main(int argc, char** argv) {
    std::string model_dir = "";
    int         port      = 8080;

    for (int i = 1; i < argc; ++i) {
        if      (std::strcmp(argv[i], "--help")  == 0) { usage(argv[0]); return 0; }
        else if (std::strcmp(argv[i], "--model") == 0 && i+1 < argc) model_dir = argv[++i];
        else if (std::strcmp(argv[i], "--port")  == 0 && i+1 < argc) port = std::atoi(argv[++i]);
    }
    if (model_dir.empty()) { usage(argv[0]); return 1; }

    const fs::path dir(model_dir);
    for (const auto& p : {dir/"config.json", dir/"model.safetensors", dir/"tokenizer.json"}) {
        if (!fs::exists(p)) { std::cerr << "Missing: " << p << "\n"; return 1; }
    }

    // ── Load once ────────────────────────────────────────────────────────────
    std::cerr << "Loading model...\n";
    ServerState st;
    st.cfg    = load_config((dir/"config.json").string());
    SafetensorsFile sf((dir/"model.safetensors").string());
    st.w_fp32 = load_weights(sf, st.cfg);
    st.rope   = build_rope_cache(st.cfg.max_position_embeddings,
                                  st.cfg.head_dim, st.cfg.rope_theta);
    st.tok    = load_tokenizer((dir/"tokenizer.json").string());

    // ── Bind + listen ────────────────────────────────────────────────────────
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { perror("socket"); return 1; }

    int yes = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);  // localhost only
    addr.sin_port        = htons(static_cast<uint16_t>(port));
    if (bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        perror("bind"); return 1;
    }
    if (listen(srv, 8) < 0) { perror("listen"); return 1; }

    std::cerr << "\nminillm server ready → http://localhost:" << port << "\n";
    std::cerr << "Press Ctrl+C to stop.\n\n";

    while (true) {
        sockaddr_in cli_addr{};
        socklen_t   cli_len  = sizeof(cli_addr);
        int sock = accept(srv, reinterpret_cast<sockaddr*>(&cli_addr), &cli_len);
        if (sock < 0) { perror("accept"); continue; }

        std::string raw = read_request(sock);
        if (raw.empty()) { close(sock); continue; }

        std::string resp = handle_request(st, raw);
        send(sock, resp.data(), resp.size(), 0);
        close(sock);
    }
    return 0;
}
