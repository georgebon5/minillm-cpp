#!/usr/bin/env python3
"""
Generate reference tokenizer outputs for round-trip testing.

Usage:
    python tools/gen_tokenizer_refs.py

Writes tools/reference_tokenizer.json with (text, ids) pairs
produced by HuggingFace AutoTokenizer for SmolLM2-135M.
"""

import json
from pathlib import Path
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).parent.parent
MODEL_DIR = REPO_ROOT / "models" / "SmolLM2-135M"
OUT_FILE  = REPO_ROOT / "tools" / "reference_tokenizer.json"

SAMPLES = [
    "Hello, world!",
    "The capital of France is",
    "def fibonacci(n):",
    "  return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "1 + 1 = 2",
    "  leading space",
    "multiple   spaces",
]

tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))

records = []
for text in SAMPLES:
    ids = tok.encode(text, add_special_tokens=False)
    records.append({"text": text, "ids": ids})

OUT_FILE.write_text(json.dumps({"samples": records}, indent=2))
print(f"Wrote {len(records)} samples to {OUT_FILE}")
