# minillm.cpp

A minimal LLM inference engine written from scratch in C++20 — no PyTorch,
ggml, ONNX, Eigen, or BLAS at runtime. CPU-only, ARM NEON optimized,
targets Apple Silicon (M2).

[![CI](https://github.com/georgebon5/minillm-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/georgebon5/minillm-cpp/actions/workflows/ci.yml)

## What it does

Loads a HuggingFace transformer checkpoint and generates text:

```bash
./build/minillm-generate \
    --model models/SmolLM2-135M \
    --prompt "The capital of France is" \
    --max-tokens 32
# → The capital of France is the capital of France. The capital of France.
# → [32 tokens, 489 ms, 65 tok/s]
```

## Features

- **Loader** — mmap-backed `safetensors` with lazy BF16 → F32 conversion
- **Model** — RMSNorm, grouped-query attention (GQA), RoPE, SwiGLU MLP,
  tied embeddings — tested with SmolLM2-135M and SmolLM2-360M
- **Tokenizer** — byte-level BPE (GPT-2 style), encode + decode from
  `tokenizer.json`
- **Inference** — KV-cache + incremental `forward_one`, greedy argmax or
  temperature + top-K sampling (seedable)
- **Performance** — hand-written NEON intrinsics (4-accumulator FMA dot
  product) with scalar fallback
- **Quantization** — symmetric **per-channel** INT8 / INT4 post-training
  quantization, nibble-packed storage
- **Web UI** (`minillm-serve`) — single-binary HTTP server with SSE token
  streaming, live top-5 token probability panel (model's "thought process"),
  side-by-side fp32/int8/int4 comparison, no external HTTP libraries
- **Testing** — 97 Catch2 tests, Ubuntu + macOS-14 CI matrix with
  ASan/UBSan, numerical correctness verified against PyTorch F32 reference

## Results on SmolLM2-135M (Apple M2, Release build)

### Speed

| | Debug (ASan) | Release (NEON) |
|---|---:|---:|
| `linear` seq=1, 576→1536 (decode step) | 3.37 ms | **0.07 ms** |
| `linear` seq=128, 576→1536 (prefill) | 377 ms | **8.20 ms** |
| End-to-end greedy decode | 1 tok/s | **61 tok/s** |

### Quantization quality (perplexity, lower is better)

Per-channel symmetric quantization, lm_head and all projection matrices
quantized, FP32 activations:

| | FP32 | INT8 | INT4 |
|---|---:|---:|---:|
| Perplexity | 76.97 | **78.46** | **625** |
| Degradation vs FP32 | — | +2.0% | +712% |
| Weight memory | 1× | **4×** | **8×** |

The INT4 result demonstrates why per-tensor quantization fails
catastrophically at 4 bits (476M PPL in an earlier commit) and
motivates per-row scales — the same insight that drives GPTQ and AWQ.

## Build and run

### Prerequisites
- CMake ≥ 3.20, C++20 compiler (clang 14+ or gcc 11+)
- Python 3.10+ with `torch`, `transformers`, `safetensors`,
  `huggingface_hub` (only for one-time weight download and reference-data
  generation)

### Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Debug builds enable AddressSanitizer and UBSan automatically.

### Download model

```bash
pip install -r tools/requirements.txt
python tools/download_model.py                                    # 135M, ~270 MB
MODEL_ID=HuggingFaceTB/SmolLM2-360M python tools/download_model.py \
    --out models/SmolLM2-360M                                     # 360M, ~720 MB
```

The 360M model produces noticeably more coherent output; the 135M is
faster and useful for correctness testing.

### Run

```bash
# Greedy text generation
./build/minillm-generate \
    --model  models/SmolLM2-135M \
    --prompt "The quick brown fox"

# Perplexity + quantization comparison
./build/minillm-perplexity \
    --model models/SmolLM2-135M \
    --text  "In machine learning, large language models..." \
    --compare

# Web UI (open http://localhost:8080)
./build/minillm-serve --model models/SmolLM2-360M
```

The web UI shows the model's live decode state: streaming tokens, memory
footprint per quant level, and a top-5 probability panel that updates
after each decode step so you can see exactly which tokens the model
considered and why temperature sampling picks what it does.

### Test

```bash
# Unit tests always pass (no model files needed)
ctest --test-dir build --output-on-failure

# To run full integration tests, generate reference data first:
python tools/export_weights.py            # F32 reference values
python tools/gen_reference_logits.py      # top-5 token ids + logits
python tools/gen_tokenizer_refs.py        # HF tokenizer samples
ctest --test-dir build
```

## Project structure

```
include/minillm/   headers (tensor, ops, model, tokenizer, safetensors)
src/               implementation + minillm-generate / minillm-perplexity
tests/             97 Catch2 tests (unit + integration vs PyTorch)
benchmarks/        bench_ops: linear and rms_norm microbench
tools/             Python scripts (model download, reference data)
.github/workflows/ CI (Ubuntu + macOS-14, ASan/UBSan)
```

## Design notes

- **Zero-copy loader.** `SafetensorsFile` mmaps the checkpoint and returns
  non-owning `Tensor` views into the mapped region. BF16 → F32 is a
  bit-exact zero-extension (`uint32_t bits = uint16_t(bf16) << 16; memcpy`).
- **KV-cache.** Each layer owns two `std::vector<float>` buffers that grow
  by one row per decode step. The attention path uses the same
  `scaled_dot_product_attention` as prefill, generalized to `q_len ≠ kv_len`.
- **NEON dot.** Four independent `float32x4_t` accumulators to break the
  loop-carried dependency and fill the M2's wide out-of-order pipeline.
- **Per-channel quantization.** One scale per output row. Different FFN
  neurons have very different weight magnitudes; a single tensor-wide scale
  wastes 4-bit codepoints on dynamic range that only the largest neuron
  needs. Per-row scales keep INT4 within an order of magnitude of FP32 PPL.

## What's intentionally simple

- Greedy decoding only (no temperature / top-p sampling)
- Pretokenizer handles single-space word boundaries, not HF's full GPT-2
  regex for grouped whitespace runs — rare for inference prompts
- No batching, no streaming output
- No INT8 KV-cache (weights only)

## License

MIT.
