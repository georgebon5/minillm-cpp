#include "minillm/model.hpp"
#include "minillm/model_config.hpp"
#include "minillm/safetensors.hpp"
#include "minillm/tokenizer.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace minillm;
namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;

static void usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "  --model      <dir>   Model directory (contains config.json,\n"
        << "                       model.safetensors, tokenizer.json)\n"
        << "  --prompt     <text>  Input prompt (default: \"Hello\")\n"
        << "  --max-tokens <n>     Maximum new tokens to generate (default: 64)\n"
        << "  --help               Show this message\n";
}

int main(int argc, char** argv) {
    std::string model_dir  = "";
    std::string prompt     = "Hello";
    int         max_tokens = 64;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0) { usage(argv[0]); return 0; }
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            model_dir = argv[++i];
        else if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc)
            prompt = argv[++i];
        else if (std::strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc)
            max_tokens = std::atoi(argv[++i]);
    }

    if (model_dir.empty()) {
        std::cerr << "Error: --model is required.\n\n";
        usage(argv[0]);
        return 1;
    }

    const fs::path dir(model_dir);
    const fs::path config_path = dir / "config.json";
    const fs::path model_path  = dir / "model.safetensors";
    const fs::path tok_path    = dir / "tokenizer.json";

    for (const auto& p : {config_path, model_path, tok_path}) {
        if (!fs::exists(p)) {
            std::cerr << "Error: file not found: " << p << "\n";
            return 1;
        }
    }

    // ── Load ─────────────────────────────────────────────────────────────────
    std::cerr << "Loading model from " << model_dir << " ...\n";

    ModelConfig     cfg  = load_config(config_path.string());
    SafetensorsFile sf(model_path.string());
    ModelWeights    w    = load_weights(sf, cfg);
    RopeCache       rope = build_rope_cache(cfg.max_position_embeddings,
                                             cfg.head_dim, cfg.rope_theta);
    Tokenizer       tok  = load_tokenizer(tok_path.string());

    std::cerr << "  vocab=" << cfg.vocab_size
              << "  layers=" << cfg.num_hidden_layers
              << "  hidden=" << cfg.hidden_size << "\n";

    // ── Tokenize prompt ───────────────────────────────────────────────────────
    auto prompt_ids = encode(tok, prompt);
    std::cerr << "Prompt (" << prompt_ids.size() << " tokens): " << prompt << "\n";
    std::cerr << "Generating up to " << max_tokens << " tokens...\n\n";

    // ── Generate ──────────────────────────────────────────────────────────────
    auto t0 = clk::now();
    auto gen_ids = generate(w, cfg, rope, prompt_ids, max_tokens, tok.eos_id);
    double elapsed_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();

    // ── Decode and print ──────────────────────────────────────────────────────
    std::string output = decode(tok, gen_ids);

    std::cout << prompt << output << "\n";

    double tps = static_cast<double>(gen_ids.size()) / (elapsed_ms / 1000.0);
    std::cerr << "\n[" << gen_ids.size() << " tokens, "
              << static_cast<int>(elapsed_ms) << " ms, "
              << static_cast<int>(tps) << " tok/s]\n";

    return 0;
}
