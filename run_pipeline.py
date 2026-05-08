#!/usr/bin/env python3
"""Top-level pipeline orchestrator.

Runs all three stages (collection, preprocessing, training) or specific stages
via the --stage flag.

Usage:
    python run_pipeline.py                 # All stages
    python run_pipeline.py --stage 1       # Collection only
    python run_pipeline.py --stage 2       # Preprocessing only
    python run_pipeline.py --stage 3       # Training only
    python run_pipeline.py --stage 2 3     # Preprocessing + Training
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="EDM Subgenre Classification Pipeline"
    )
    parser.add_argument(
        "--stage",
        type=int,
        nargs="+",
        choices=[1, 2, 3],
        default=[1, 2, 3],
        help="Stage(s) to run: 1=collection, 2=preprocessing, 3=training (default: all)",
    )
    args = parser.parse_args()
    stages = sorted(set(args.stage))

    if 1 in stages:
        print("\n" + "=" * 60)
        print("STAGE 1: Data Collection")
        print("=" * 60)
        from data_collection.run_collection import run as run_collection
        run_collection()

    if 2 in stages:
        print("\n" + "=" * 60)
        print("STAGE 2: Preprocessing")
        print("=" * 60)
        from preprocessing.run_preprocessing import run as run_preprocessing
        run_preprocessing()

    if 3 in stages:
        print("\n" + "=" * 60)
        print("STAGE 3: Model Training")
        print("=" * 60)
        from model_training.run_training import run as run_training
        run_training()

    print("\n" + "=" * 60)
    print(f"Pipeline complete (stages: {stages})")
    print("=" * 60)


if __name__ == "__main__":
    main()
