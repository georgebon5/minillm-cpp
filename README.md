# minillm.cpp

A from-scratch LLM inference engine in C++20. No PyTorch, no ggml, no Eigen, no BLAS — just the math, a POSIX socket, and a weekend that turned into three weeks.

[![CI](https://github.com/georgebon5/minillm-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/georgebon5/minillm-cpp/actions/workflows/ci.yml)

---

I built this because I kept using tools like llama.cpp and transformers without really understanding what was happening inside them. This project is my attempt to close that gap — implement everything myself, verify it against PyTorch reference values, and see how fast I can make it on Apple Silicon.

It runs SmolLM2 (135M and 360M) on CPU at 60+ tok/s in release mode.

## What it does

```bash
./build/minillm-generate \
    --model models/SmolLM2-360M \
    --prompt "The key idea behind transformers is" \
    --max-tokens 64
```

There's also a web UI:

```bash
./build/minillm-serve --model models/SmolLM2-360M
# open http://localhost:8080
```

The UI streams tokens live, shows a per-token top-5 probability panel so you can see what the model was considering at each step, and has a side-by-side comparison mode that runs FP32 / INT8 / INT4 in sequence so you can see where quantization degrades output.

## Architecture

The transformer stack is standard: RMSNorm → GQA → RoPE → SwiGLU MLP, with a KV cache for efficient decode. Everything I implemented I verified numerically — the test suite runs forward passes and checks logits against PyTorch F32 reference values to within floating-point tolerance.

**Loader** — `SafetensorsFile` mmaps the checkpoint and hands out non-owning tensor views. BF16 → F32 conversion is a zero-extension (`uint32_t bits = uint16_t(bf16) << 16; memcpy`) — bit-exact and fast.

**Attention** — Grouped-query attention (GQA) with RoPE. During prefill the whole sequence goes through `forward()` at once; during decode `forward_one()` appends one K/V row per layer and attends over the full cache. No causal mask needed on single-query decode.

**NEON dot product** — Four independent `float32x4_t` accumulators break the loop-carried dependency and keep the M2's out-of-order pipeline fed. Scalar fallback for non-ARM builds. This alone is the difference between 8 tok/s and 60 tok/s.

**Quantization** — Post-training symmetric per-channel INT8 and INT4. One scale per output row. Per-row matters a lot at 4 bits: an earlier version used a single per-tensor scale and hit 476k perplexity. Switching to per-channel dropped that to 625. Still bad, but the degradation is now predictable and explainable — and it's the same insight that drives GPTQ and AWQ.

**Sampling** — Temperature + top-K + top-p (nucleus) + repetition penalty. Greedy decoding loops badly on 135M, especially with repetition. The penalty helps a lot once you dial it in.

**Server** — Single-binary HTTP server over raw POSIX sockets, no external HTTP library. SSE for live token streaming. The UTF-8 handling was annoying: byte-level BPE can split a multi-byte character across tokens, so the SSE path buffers incomplete codepoints and only flushes whole characters.

## Performance (Apple M2, SmolLM2-135M, Release build)

| Operation | Debug (ASan) | Release (NEON) |
|---|---:|---:|
| `linear` seq=1, 576→1536 | 3.37 ms | **0.07 ms** |
| `linear` seq=128, 576→1536 | 377 ms | **8.20 ms** |
| End-to-end decode | 1 tok/s | **61 tok/s** |

## Quantization quality (SmolLM2-135M, perplexity — lower is better)

Per-channel symmetric, lm_head and all projection matrices quantized, FP32 activations.

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
python tools/download_model.py                              # 135M, ~270 MB
MODEL_ID=HuggingFaceTB/SmolLM2-360M \
    python tools/download_model.py --out models/SmolLM2-360M  # 360M, ~720 MB
```

The 360M model is noticeably more coherent. The 135M is faster and useful for testing.

## Run

```bash
# CLI generation
./build/minillm-generate \
    --model models/SmolLM2-360M \
    --prompt "Attention mechanisms work by"

# Perplexity + quant comparison
./build/minillm-perplexity \
    --model models/SmolLM2-135M \
    --text "In machine learning, large language models..." \
    --compare

# Web UI
./build/minillm-serve --model models/SmolLM2-360M
```

The server does a RAM check before loading — it estimates the F32 footprint from the on-disk BF16 size and aborts if loading would push past 75% of physical memory. Learned this the hard way after the 1.7B model turned a 16 GB machine into a page-swap nightmare.

## Test

```bash
# Unit tests (no model files needed)
ctest --test-dir build --output-on-failure

# Integration tests (requires model + reference data)
python tools/export_weights.py
python tools/gen_reference_logits.py
python tools/gen_tokenizer_refs.py
ctest --test-dir build
```

97 test cases across ops, tensor, quantization, tokenizer, loader, and full forward pass.

## Structure

```
include/minillm/   tensor, ops, model, tokenizer, safetensors headers
src/               implementations + three binaries (generate, perplexity, serve)
tests/             97 Catch2 tests
benchmarks/        linear + rms_norm microbenchmarks
tools/             Python scripts for model download and reference data
.github/workflows/ CI matrix: Ubuntu + macOS-14, ASan/UBSan
```

## What I cut

- No batching. One request at a time.
- No INT8 KV cache — weights only.
- The pretokenizer handles simple word boundaries, not the full GPT-2 regex for grouped whitespace. Close enough for inference prompts.
- No Flash Attention. The naive O(n²) attention is fine at these sequence lengths.

## License

MIT.
