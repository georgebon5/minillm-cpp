#!/usr/bin/env python3
"""
Run the SmolLM2-135M forward pass in PyTorch (float32) and save the last-token
logits as a reference for the C++ correctness test (Task 2.10).

Usage:
    python tools/gen_reference_logits.py
    python tools/gen_reference_logits.py --model-dir models/SmolLM2-135M --out tools/reference_logits.json
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TEST_TEXT = "The capital of France is"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/SmolLM2-135M")
    parser.add_argument("--out", default="tools/reference_logits.json")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"ERROR: {model_dir} not found. Run download_model.py first.")

    print(f"Loading model from {model_dir} in float32 ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.float32
    )
    model.eval()

    input_ids = tokenizer.encode(TEST_TEXT, return_tensors="pt")
    print(f"Input: {TEST_TEXT!r}")
    print(f"Token IDs: {input_ids[0].tolist()}")

    with torch.no_grad():
        logits_all = model(input_ids).logits[0]          # [seq_len, vocab]
        last_logits = logits_all[-1].tolist()             # [vocab]
        top5 = torch.topk(logits_all[-1], 5)

    result = {
        "text":              TEST_TEXT,
        "input_ids":         input_ids[0].tolist(),
        "last_token_logits": last_logits,
        "top5_token_ids":    top5.indices.tolist(),
        "top5_values":       top5.values.tolist(),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fp:
        json.dump(result, fp, indent=2)

    print(f"\nTop-5 predicted tokens:")
    for tid, val in zip(result["top5_token_ids"], result["top5_values"]):
        print(f"  {tokenizer.decode([tid])!r:15s}  logit={val:.4f}")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
