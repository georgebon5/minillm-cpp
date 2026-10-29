#!/usr/bin/env python3
"""
Load model.safetensors and write a reference JSON with tensor shapes and
first-4 F32 values. The C++ loader test (Task 1.8) uses this file to verify
correctness without needing PyTorch at test time.

Usage:
    python tools/export_weights.py
    python tools/export_weights.py --model-dir models/SmolLM2-135M --out tools/reference_tensors.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import safe_open as safe_open_torch


def to_f32_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor of any dtype to a float32 numpy array."""
    return t.to(torch.float32).contiguous().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/SmolLM2-135M",
                        help="Directory containing model.safetensors")
    parser.add_argument("--out", default="tools/reference_tensors.json",
                        help="Output JSON path")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    st_path = model_dir / "model.safetensors"

    if not st_path.exists():
        raise SystemExit(
            f"ERROR: {st_path} not found.\n"
            "Run `python tools/download_model.py` first."
        )

    print(f"Reading {st_path} …")
    reference: dict = {}

    with safe_open_torch(str(st_path), framework="pt") as f:
        for name in sorted(f.keys()):
            t       = f.get_tensor(name)
            arr_f32 = to_f32_numpy(t)
            reference[name] = {
                "shape":   list(t.shape),
                "dtype":   str(t.dtype).replace("torch.", ""),
                "first_4": arr_f32.flat[:4].tolist(),
            }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fp:
        json.dump(reference, fp, indent=2)

    print(f"Saved reference for {len(reference)} tensors → {out}\n")
    print(f"{'Tensor name':<55} {'Shape':<22} First 4 values (F32)")
    print("-" * 100)
    for name, info in reference.items():
        vals = [f"{v:.6f}" for v in info["first_4"]]
        print(f"{name:<55} {str(info['shape']):<22} {vals}")


if __name__ == "__main__":
    main()
