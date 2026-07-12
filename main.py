#!/usr/bin/env python3
"""CyberDetect AI — command-line interface.

A small, self-contained network intrusion detection system built on an
Isolation Forest anomaly detector.

Examples
--------
  # 1. Create a synthetic traffic dataset
  python main.py generate --samples 20000 --attack-ratio 0.08

  # 2. Train the detector and print evaluation metrics
  python main.py train

  # 3. Scan a CSV of flows and list the most suspicious ones
  python main.py detect --input data/network_traffic.csv --top 15

  # 4. End-to-end demo (generate -> train -> detect) in one command
  python main.py demo
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.config import DATASET_PATH
from src.data_generator import generate_dataset
from src.detect import detect_from_csv, summarize
from src.train import train


def cmd_generate(args: argparse.Namespace) -> None:
    df = generate_dataset(n_samples=args.samples, attack_ratio=args.attack_ratio)
    df.to_csv(DATASET_PATH, index=False)
    attacks = int(df["label"].sum())
    print(f"Generated {len(df):,} flows -> {DATASET_PATH}")
    print(f"  normal : {len(df) - attacks:,}")
    print(f"  attacks: {attacks:,}")


def cmd_train(args: argparse.Namespace) -> None:
    print("Training Isolation Forest intrusion detector ...")
    _, metrics = train(
        n_samples=args.samples,
        attack_ratio=args.attack_ratio,
        test_size=args.test_size,
    )
    print()
    print(metrics.pretty())
    print("Model saved to models/ (isolation_forest.joblib, scaler.joblib)")


def cmd_detect(args: argparse.Namespace) -> None:
    result = detect_from_csv(args.input, top=args.top)
    print(summarize(result))
    print()

    display_cols = [
        "duration", "src_bytes", "dst_bytes", "packet_rate",
        "failed_logins", "unique_ports", "anomaly_score", "prediction",
    ]
    display_cols = [c for c in display_cols if c in result.columns]
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(result[display_cols].to_string(index=False))


def cmd_demo(args: argparse.Namespace) -> None:
    print("=" * 60)
    print(" CyberDetect AI — end-to-end demo")
    print("=" * 60)
    cmd_generate(args)
    print()
    cmd_train(args)
    print()
    print("Top suspicious flows from the generated dataset:")
    print("-" * 60)
    args.input = str(DATASET_PATH)
    cmd_detect(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cyberdetect",
        description="Network Intrusion Detection using Isolation Forest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--samples", type=int, default=10_000,
                        help="number of synthetic flows (default: 10000)")
    common.add_argument("--attack-ratio", type=float, default=0.08,
                        help="fraction of flows that are attacks (default: 0.08)")

    p_gen = sub.add_parser("generate", parents=[common],
                           help="generate a synthetic traffic dataset")
    p_gen.set_defaults(func=cmd_generate)

    p_train = sub.add_parser("train", parents=[common],
                             help="train the detector and report metrics")
    p_train.add_argument("--test-size", type=float, default=0.3,
                         help="held-out test fraction (default: 0.3)")
    p_train.set_defaults(func=cmd_train)

    p_det = sub.add_parser("detect", help="scan a CSV of flows for intrusions")
    p_det.add_argument("--input", required=True, help="path to a flows CSV")
    p_det.add_argument("--top", type=int, default=20,
                       help="show only the N most suspicious flows (default: 20)")
    p_det.set_defaults(func=cmd_detect)

    p_demo = sub.add_parser("demo", parents=[common],
                            help="run generate -> train -> detect end to end")
    p_demo.add_argument("--test-size", type=float, default=0.3)
    p_demo.add_argument("--top", type=int, default=15)
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
