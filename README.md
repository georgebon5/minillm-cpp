# minillm.cpp

A minimal LLM inference engine written from scratch in C++20.  
No ML libraries at runtime. CPU-only. Targets ARM NEON (Apple M2).

## Models
- SmolLM2-135M (HuggingFaceTB)
- Llama 3.2 1B *(planned)*

## Milestones
- [ ] Weight loading (safetensors + mmap)
- [ ] FP32 forward pass (embeddings, RMSNorm, RoPE, GQA, SwiGLU)
- [ ] BPE tokenizer
- [ ] Generation loop + KV-cache
- [ ] NEON SIMD performance optimizations
- [ ] INT8 / INT4 weight-only quantization
- [ ] Evaluation vs llama.cpp

## Build
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build
```

## Benchmarks
*To be filled after Milestone 5.*

## Perplexity (WikiText-2)
*To be filled after Milestone 6.*
