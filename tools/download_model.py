#!/usr/bin/env python3
"""Download SmolLM2 weights and config from HuggingFace Hub.

By default downloads SmolLM2-135M (base). Pass --instruct to get the
instruction-tuned variant, or set MODEL_ID directly for any other model.

Examples:
    python tools/download_model.py
    python tools/download_model.py --instruct
    python tools/download_model.py --size 360M --instruct
    python tools/download_model.py --size 1.7B --instruct --out models/SmolLM2-1.7B-Instruct
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download

REQUIRED_FILES = [
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", default="135M", choices=["135M", "360M", "1.7B"],
                        help="Model size (default: 135M)")
    parser.add_argument("--instruct", action="store_true",
                        help="Download the Instruct (chat-tuned) variant")
    parser.add_argument("--out", default=None,
                        help="Destination directory (auto-derived from size/variant if omitted)")
    args = parser.parse_args()

    # Allow full override via env var (backwards-compatible with old usage)
    model_id = os.environ.get("MODEL_ID")
    if model_id is None:
        suffix = "-Instruct" if args.instruct else ""
        model_id = f"HuggingFaceTB/SmolLM2-{args.size}{suffix}"

    out = Path(args.out) if args.out else Path(f"models/SmolLM2-{args.size}{'-Instruct' if args.instruct else ''}")
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_id} → {out}/")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(out),
        allow_patterns=REQUIRED_FILES,
    )

    print("\nVerifying files:")
    all_ok = True
    for name in REQUIRED_FILES:
        path = out / name
        if path.exists():
            print(f"  OK  {name}  ({path.stat().st_size:,} bytes)")
        else:
            print(f"  MISSING  {name}")
            all_ok = False

    if not all_ok:
        raise SystemExit("Some files are missing — re-run the script.")

    print(f"\nDone. Run the server with:")
    print(f"  ./build/minillm-serve --model {out}")


if __name__ == "__main__":
    main()
