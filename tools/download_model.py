#!/usr/bin/env python3
"""Download SmolLM2-135M weights and config from HuggingFace Hub."""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

import os
MODEL_ID = os.environ.get("MODEL_ID", "HuggingFaceTB/SmolLM2-135M")

REQUIRED_FILES = [
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="models/SmolLM2-135M",
        help="Destination directory (default: models/SmolLM2-135M)",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {MODEL_ID} → {out}/")
    snapshot_download(
        repo_id=MODEL_ID,
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


if __name__ == "__main__":
    main()
