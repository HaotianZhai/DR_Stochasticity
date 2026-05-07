#!/usr/bin/env python3
"""
Download WebWalkerQA dataset from HuggingFace and convert to evaluation format.
https://huggingface.co/datasets/callanwu/WebWalkerQA
"""

import json
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert WebWalkerQA dataset"
    )
    parser.add_argument(
        "--num-instances", type=int, default=100,
        help="Number of instances to extract (default: 100, use -1 for all)"
    )
    parser.add_argument(
        "--output", type=str, default="data/webwalkerqa.json",
        help="Output file path (default: data/webwalkerqa.json)"
    )
    parser.add_argument(
        "--split", type=str, default="main",
        choices=["main", "silver"],
        help="Dataset split (default: main, 680 human-verified questions)"
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed. Install with: pip install datasets")
        return

    print("Loading WebWalkerQA dataset from HuggingFace...")
    dataset = load_dataset("callanwu/WebWalkerQA", split=args.split)
    print(f"Dataset loaded: {len(dataset)} total examples (split={args.split})")

    limit = len(dataset) if args.num_instances == -1 else min(args.num_instances, len(dataset))
    print(f"Extracting {limit} instances...")

    converted_data = []
    for i, example in enumerate(dataset):
        if i >= limit:
            break
        converted_data.append({
            "question": example["question"],
            "answer": example["answer"]
        })

    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(converted_data)} instances to {args.output}")

    print("\nFirst 3 examples:")
    for i, item in enumerate(converted_data[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Q: {item['question'][:200]}...")
        print(f"A: {item['answer'][:100]}...")


if __name__ == "__main__":
    main()
