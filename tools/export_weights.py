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
from safetensors import safe_open


def bf16_bytes_to_f32(raw: bytes) -> np.ndarray:
    """Reinterpret raw BF16 bytes as F32 by zero-extending each 2-byte value."""
    u16 = np.frombuffer(raw, dtype="<u2")
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


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

    with safe_open(str(st_path), framework="numpy") as f:
        for name in sorted(f.keys()):
            arr = f.get_tensor(name)           # numpy array, original dtype

            # Convert to F32 regardless of stored dtype.
            if str(arr.dtype) == "bfloat16":
                # numpy has no native bfloat16; reinterpret bytes manually.
                arr_f32 = bf16_bytes_to_f32(arr.tobytes()).reshape(arr.shape)
            else:
                arr_f32 = arr.astype(np.float32)

            reference[name] = {
                "shape":   list(arr.shape),
                "dtype":   str(arr.dtype),
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
