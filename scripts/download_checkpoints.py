#!/usr/bin/env python3
"""
EfficientSAM3 Checkpoint Downloader via Hugging Face Hub.
Downloads model weights directly into the Hugging Face cache directory ($HF_HOME/hub)
and optionally creates local symlinks or copies into a target directory (e.g. checkpoints/efficientsam3_ft).
"""

import argparse
import os
import sys

from huggingface_hub import hf_hub_download

REPO_ID = "Simon7108528/EfficientSAM3"

MODELS = {
    "tinyvit": "efficientsam3_ft/efficientsam3_tinyvit.pt",
    "efficientvit": "efficientsam3_ft/efficientsam3_efficientvit.pt",
    "repvit": "efficientsam3_ft/efficientsam3_repvit.pt",
}


def main():
    parser = argparse.ArgumentParser(description="Download EfficientSAM3 weights from Hugging Face Hub.")
    parser.add_argument(
        "--dest-dir",
        default="checkpoints/efficientsam3_ft",
        help="Local directory to place symlinks/copies (default: checkpoints/efficientsam3_ft)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["all", "tinyvit", "efficientvit", "repvit"],
        default=["all"],
        help="Models to download (default: all)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Custom HF cache directory (defaults to HF_HOME/hub or ~/.cache/huggingface/hub)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy file into dest-dir instead of symlinking",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if already cached",
    )
    args = parser.parse_args()

    selected = list(MODELS.keys()) if "all" in args.models else args.models
    os.makedirs(args.dest_dir, exist_ok=True)

    print("=" * 60)
    print("EfficientSAM3 Hugging Face Checkpoint Downloader")
    print(f"Repository: {REPO_ID}")
    print(f"Destination: {args.dest_dir}")
    print("=" * 60)

    for m in selected:
        hf_filename = MODELS[m]
        base_name = os.path.basename(hf_filename)
        dest_path = os.path.join(args.dest_dir, base_name)
        print(f"[*] Downloading {m} ({hf_filename})...")
        cached_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=hf_filename,
            cache_dir=args.cache_dir,
            force_download=args.force,
        )
        print(f"    Cached at: {cached_path}")
        if not os.path.exists(dest_path):
            try:
                if args.copy:
                    import shutil

                    shutil.copyfile(cached_path, dest_path)
                    print(f"    Copied to: {dest_path}")
                else:
                    os.symlink(os.path.abspath(cached_path), dest_path)
                    print(f"    Symlinked to: {dest_path}")
            except Exception as e:
                print(f"    Warning: Could not link to {dest_path}: {e}")
        else:
            print(f"    [OK] Local destination exists: {dest_path}")

    print("=" * 60)
    print("All requested checkpoints are ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
