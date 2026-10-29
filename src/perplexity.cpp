#include "minillm/model.hpp"
#include "minillm/model_config.hpp"
#include "minillm/ops.hpp"
#include "minillm/safetensors.hpp"
#include "minillm/tokenizer.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace minillm;
namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;

static void usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "  --model   <dir>          Model directory\n"
        << "  --text    <file|-|str>   Text file, '-' for stdin, or literal string\n"
        << "  --quant   fp32|int8|int4 Quantization level (default: fp32)\n"
        << "  --chunk   <n>            Tokens per forward chunk (default: 256)\n"
        << "  --compare                Run fp32 + int8 + int4 and print comparison\n"
        << "  --help                   Show this message\n";
}

// Read text from file path, stdin ('-'), or treat the argument as a literal string.
static std::string read_text(const std::string& src) {
    if (src == "-") {
        return {std::istreambuf_iterator<char>(std::cin),
                std::istreambuf_iterator<char>()};
    }
    if (fs::exists(src)) {
        std::ifstream f(src);
        return {std::istreambuf_iterator<char>(f),
                std::istreambuf_iterator<char>()};
    }
    return src;  // treat as literal string
}

// Numerically-stable log-sum-exp over a row of `n` floats.
static float log_sum_exp(const float* row, size_t n) {
    float mx = *std::max_element(row, row + n);
    float s  = 0.f;
    for (size_t i = 0; i < n; ++i) s += std::exp(row[i] - mx);
    return std::log(s) + mx;
}

struct PPLResult {
    float  ppl;
    double ms;
    int    n_tokens;
};

static PPLResult compute_ppl(const ModelWeights& w,
                               const ModelConfig&  cfg,
                               const RopeCache&    rope,
                               const std::vector<int32_t>& tokens,
                               int chunk_size) {
    if (tokens.size() < 2) return {0.f, 0.0, 0};

    double nll     = 0.0;
    int    counted = 0;

    auto t0 = clk::now();

    for (size_t start = 0; start + 1 < tokens.size(); start += static_cast<size_t>(chunk_size)) {
        size_t end   = std::min(start + static_cast<size_t>(chunk_size), tokens.size());
        std::span<const int32_t> chunk(tokens.data() + start, end - start);

        Tensor logits = forward(w, cfg, rope, chunk);  // [chunk_len, vocab]

        const float* lp    = logits.data_as<float>();
        const size_t vocab = logits.shape()[1];

        // For each position i, predict token[start+i+1].
        for (size_t i = 0; i + 1 < chunk.size(); ++i) {
            const float* row     = lp + i * vocab;
            int32_t      next_id = chunk[i + 1];
            float        lse     = log_sum_exp(row, vocab);
            nll += static_cast<double>(lse - row[static_cast<size_t>(next_id)]);
            ++counted;
        }
    }

    double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    float  ppl = static_cast<float>(std::exp(nll / counted));
    return {ppl, ms, counted};
}

int main(int argc, char** argv) {
    std::string model_dir   = "";
    std::string text_src    = "The quick brown fox jumps over the lazy dog.";
    std::string quant_str   = "fp32";
    int         chunk_size  = 256;
    bool        do_compare  = false;

    for (int i = 1; i < argc; ++i) {
        if      (std::strcmp(argv[i], "--help")    == 0) { usage(argv[0]); return 0; }
        else if (std::strcmp(argv[i], "--model")   == 0 && i+1 < argc) model_dir  = argv[++i];
        else if (std::strcmp(argv[i], "--text")    == 0 && i+1 < argc) text_src   = argv[++i];
        else if (std::strcmp(argv[i], "--quant")   == 0 && i+1 < argc) quant_str  = argv[++i];
        else if (std::strcmp(argv[i], "--chunk")   == 0 && i+1 < argc) chunk_size = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--compare") == 0)                do_compare = true;
    }

    if (model_dir.empty()) { usage(argv[0]); return 1; }

    const fs::path dir(model_dir);
    for (const auto& p : {dir/"config.json", dir/"model.safetensors", dir/"tokenizer.json"}) {
        if (!fs::exists(p)) { std::cerr << "Missing: " << p << "\n"; return 1; }
    }

    // ── Load ─────────────────────────────────────────────────────────────────
    std::cerr << "Loading model...\n";
    ModelConfig     cfg  = load_config((dir/"config.json").string());
    SafetensorsFile sf((dir/"model.safetensors").string());
    ModelWeights    w    = load_weights(sf, cfg);
    RopeCache       rope = build_rope_cache(cfg.max_position_embeddings,
                                             cfg.head_dim, cfg.rope_theta);
    Tokenizer       tok  = load_tokenizer((dir/"tokenizer.json").string());

    // ── Tokenize ─────────────────────────────────────────────────────────────
    std::string text = read_text(text_src);
    auto tokens = encode(tok, text);
    std::cerr << "Text: " << tokens.size() << " tokens\n\n";

    if (tokens.size() < 2) {
        std::cerr << "Error: need at least 2 tokens to compute perplexity.\n";
        return 1;
    }

    // ── Evaluate ─────────────────────────────────────────────────────────────
    auto print_row = [&](const std::string& label, const ModelWeights& weights) {
        auto r = compute_ppl(weights, cfg, rope, tokens, chunk_size);
        double tps = r.n_tokens / (r.ms / 1000.0);
        std::printf("  %-6s  PPL = %7.2f   %5d tokens  %6.0f ms  %5.0f tok/s\n",
                    label.c_str(), r.ppl, r.n_tokens,
                    r.ms, tps);
        std::fflush(stdout);
    };

    std::printf("%-6s  %-13s  %-12s  %-8s  %s\n",
                "quant", "perplexity", "n_tokens", "time", "tok/s");
    std::printf("%s\n", std::string(60, '-').c_str());

    if (do_compare) {
        print_row("fp32",  w);
        std::cerr << "Quantizing INT8...\n";
        print_row("int8",  quantize_weights(w, DType::I8));
        std::cerr << "Quantizing INT4...\n";
        print_row("int4",  quantize_weights(w, DType::I4));
    } else {
        DType qdtype = DType::F32;
        if      (quant_str == "int8") qdtype = DType::I8;
        else if (quant_str == "int4") qdtype = DType::I4;

        if (qdtype != DType::F32) {
            std::cerr << "Quantizing weights to " << quant_str << "...\n";
            auto wq = quantize_weights(w, qdtype);
            print_row(quant_str, wq);
        } else {
            print_row("fp32", w);
        }
    }

    return 0;
}
