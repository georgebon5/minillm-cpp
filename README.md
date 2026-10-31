# minillm.cpp

A from-scratch LLM inference engine in C++20. No PyTorch, no ggml, no Eigen, no BLAS — just the math, a POSIX socket, and a weekend that turned into three weeks.

[![CI](https://github.com/georgebon5/minillm-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/georgebon5/minillm-cpp/actions/workflows/ci.yml)

**[→ Live demo](https://unwashed-swampland-chaffing.ngrok-free.dev)** *(running SmolLM2-1.7B-Instruct on a local MacBook — may be offline outside working hours)*

---

I built this because I kept using tools like llama.cpp and transformers without really understanding what was happening inside them. This project is my attempt to close that gap — implement everything myself, verify it against PyTorch reference values, and see how fast I can make it on Apple Silicon.

It runs SmolLM2 (135M, 360M, and 1.7B) on CPU. The 1.7B model fits in 8 GB RAM thanks to a direct BF16→INT8 loading path that never materialises a full FP32 copy.

## What it does

```bash
./build/minillm-generate \
    --model models/SmolLM2-360M \
    --prompt "The key idea behind transformers is" \
    --max-tokens 64
```

There's also a web UI:

```bash
./build/minillm-serve --model models/SmolLM2-1.7B-Instruct --quant int8
# open http://localhost:8080
```

The UI streams tokens live, shows a per-token top-5 probability panel so you can see what the model was considering at each step, and has a side-by-side comparison mode that runs FP32 / INT8 / INT4 in sequence so you can see where quantization starts to hurt.

## Architecture

The transformer stack is standard: RMSNorm → GQA → RoPE → SwiGLU MLP, with a KV cache for efficient decode. Everything I implemented I verified numerically — the test suite runs forward passes and checks logits against PyTorch F32 reference values to within floating-point tolerance.

**Loader** — `SafetensorsFile` mmaps the checkpoint and hands out non-owning tensor views. BF16 → F32 conversion is a zero-extension (`uint32_t bits = uint16_t(bf16) << 16; memcpy`) — bit-exact and fast.

**Direct INT8 loading** — `load_weights_q()` reads the mmap'd BF16 file and quantizes each weight matrix row-by-row, keeping only an O(cols) temporary buffer in RAM at any point. This means the 1.7B model's peak RAM during load is ~1.9 GB instead of the ~7 GB you'd get from load-then-quantize. I learned the hard way that "simulate quantization on the FP32 weights" doesn't actually save any memory — the real trick is never loading FP32 in the first place.

**Attention** — Grouped-query attention (GQA) with RoPE. During prefill the whole sequence goes through `forward()` at once; during decode `forward_one()` appends one K/V row per layer and attends over the full cache. No causal mask needed for single-query decode.

**NEON kernels** — Two hand-written SIMD paths. The FP32 dot product uses four independent `float32x4_t` accumulators to break the loop-carried dependency. The INT8 path fuses dequantize + dot in a single pass: `vmovl_s8` widens 16 INT8 values to INT32, `vcvtq_f32_s32` converts to FP32, then `vfmaq_f32` accumulates — no temporary FP32 buffer written. This alone doubled the INT8 throughput vs the naive dequantize-then-dot approach.

**Quantization** — Post-training symmetric per-channel INT8 and INT4. One scale per output row. Per-row matters a lot at 4 bits: an earlier version used a single per-tensor scale and hit 476k perplexity on the validation set. Per-channel dropped that to 625. Still not great, but the degradation is now predictable and explainable — and it's the same insight that drives GPTQ and AWQ.

**Sampling** — Temperature + top-K + top-p (nucleus) + repetition penalty. Greedy decoding loops badly on smaller models. The penalty helps a lot once you dial it in; too high and it starts avoiding reasonable words it's already said once.

**Server** — Single-binary HTTP server over raw POSIX sockets, no external HTTP library. SSE for live token streaming. The UTF-8 handling was annoying: byte-level BPE can split a multi-byte character across tokens, so the SSE path buffers incomplete codepoints and only flushes whole characters to the client.

**Chat template** — The tokenizer detects `<|im_start|>` / `<|im_end|>` special tokens at parse time and builds a sorted split list so `encode()` can insert them directly by ID, bypassing byte-level BPE. The server wraps prompts in ChatML automatically for Instruct models and is a no-op for base models.

## Performance (Apple M2, Release build)

| Model | Quant | tok/s |
|---|---|---:|
| SmolLM2-135M | FP32 | **61** |
| SmolLM2-360M | FP32 | **36** |
| SmolLM2-1.7B | INT8 | **~10** |

*INT8 decode uses the fused NEON kernel; FP32 uses the four-accumulator dot product.*

### Linear layer microbenchmarks (135M, seq=1)

| | Debug (ASan) | Release (NEON) |
|---|---:|---:|
| `linear` 576→1536 | 3.37 ms | **0.07 ms** |
| `linear` seq=128, 576→1536 | 377 ms | **8.20 ms** |

## Quantization quality (SmolLM2-135M, perplexity — lower is better)

Per-channel symmetric, all projection matrices quantized, FP32 activations.

| | FP32 | INT8 | INT4 |
|---|---:|---:|---:|
| Perplexity | 76.97 | **78.46** | **625** |
| vs FP32 | — | +2.0% | +712% |
| Weight memory | 1× | 0.25× | 0.125× |

INT8 is nearly free quality-wise. INT4 at per-channel is survivable; per-tensor INT4 is not.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Debug builds enable AddressSanitizer and UBSan automatically.

Prerequisites: CMake ≥ 3.20, C++20 compiler (clang 14+ or gcc 11+). Python only needed for weight download and reference data generation.

## Download models

```bash
pip install -r tools/requirements.txt

# Base models (text completion)
python tools/download_model.py --size 135M          # ~270 MB
python tools/download_model.py --size 360M          # ~720 MB

# Instruct models (chat)
python tools/download_model.py --size 360M --instruct   # ~720 MB
python tools/download_model.py --size 1.7B --instruct   # ~3.4 GB on disk, ~1.9 GB in RAM with --quant int8
```

The `--instruct` flag downloads the RLHF fine-tuned variant. The server auto-detects it from `tokenizer_config.json` and applies the ChatML template automatically.

## Run

```bash
# CLI generation
./build/minillm-generate \
    --model models/SmolLM2-360M \
    --prompt "Attention mechanisms work by"

# Web UI — 360M Instruct (fast, fits anywhere)
./build/minillm-serve --model models/SmolLM2-360M-Instruct

# Web UI — 1.7B Instruct on 8 GB RAM (direct INT8 load, no FP32 peak)
./build/minillm-serve --model models/SmolLM2-1.7B-Instruct --quant int8

# Perplexity + quant comparison
./build/minillm-perplexity --model models/SmolLM2-135M --text "..." --compare
```

The server does a RAM check before loading — it estimates the footprint from the on-disk size and aborts if it would push past 85% of physical memory. Learned this the hard way after the 1.7B model turned an 8 GB machine into a page-swap nightmare.

## API

The server exposes a small JSON API alongside the web UI.

### `GET /info`
Returns metadata for all loaded models including memory footprint estimates per quantization level.

### `GET /health`
```json
{"status": "ok", "uptime_s": 42, "models": 2, "total_requests": 7, "total_tokens": 312}
```

### `POST /generate`
Blocking generation.
```json
// request
{"prompt": "Hello", "max_tokens": 64, "temperature": 0.2, "quant": "int8",
 "model": "SmolLM2-1.7B-Instruct", "system_prompt": "You are a helpful assistant."}

// response
{"generated": "Hello! How can I help you?", "n_tokens": 8, "ms": 820, "tok_per_sec": 10}
```

### `GET /stream`
Server-Sent Events. Each `token` event carries the text delta plus top-5 softmax probabilities for that position.

### `POST /tokenize`
Returns token count for a string — useful for checking context budget.

## Test

```bash
ctest --test-dir build --output-on-failure
```

124 test cases across ops, tensors, quantization, tokenizer (including ChatML boundaries), safetensors loader, and full forward pass. Integration tests verify logits against PyTorch reference values.

## Structure

```
include/minillm/   tensor, ops, model, tokenizer, safetensors headers
src/               implementations + three binaries (generate, perplexity, serve)
tests/             Catch2 test suite
benchmarks/        linear + rms_norm microbenchmarks
tools/             Python scripts for model download and reference data
.github/workflows/ CI matrix: Ubuntu + macOS-14, ASan/UBSan
```

## What I didn't implement

- No batching. One request at a time.
- No INT8 KV cache — weights are quantized, activations stay FP32.
- The pretokenizer handles simple word boundaries, not the full GPT-2 regex. Close enough for inference prompts.
- No Flash Attention. Naive O(n²) is fine at these sequence lengths.

## License

MIT.
