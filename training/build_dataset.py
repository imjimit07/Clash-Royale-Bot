#!/usr/bin/env python3
"""Build a supervised ML dataset from recorded fight packs."""
from __future__ import annotations

import argparse
from pathlib import Path

from clbot.bot.ml.dataset import iter_pack_samples, save_numpy_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("recordings", type=Path, help="Folder containing one subfolder per recorded fight")
    parser.add_argument("output", type=Path, help="Output dataset directory")
    parser.add_argument(
        "--sources",
        default="",
        help="Optional comma-separated policy sources to include, e.g. human,tactical,heuristic",
    )
    args = parser.parse_args()
    default_sources = {"human", "tactical", "heuristic", "legacy"}
    sources = {item.strip() for item in args.sources.split(",") if item.strip()} if args.sources else default_sources
    samples, encoder = iter_pack_samples(args.recordings, policy_sources=sources)
    save_numpy_dataset(samples, args.output, encoder)
    print(f"Built {len(samples)} decision samples (feature_dim={encoder.dim}).")
    if not samples:
        print("No trainable samples were found. Record new fights with the updated recorder first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
