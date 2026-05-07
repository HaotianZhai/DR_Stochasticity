#!/usr/bin/env python3
"""
Download DeepSearchQA dataset from HuggingFace and convert to evaluation format.
https://huggingface.co/datasets/google/deepsearchqa
"""

import json
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert DeepSearchQA dataset"
    )
    parser.add_argument(
        "--num-instances", type=int, default=25,
        help="Number of instances to extract (default: 25, use -1 for all 900)"
    )
    parser.add_argument(
        "--output", type=str, default="data/deepsearchqa.json",
        help="Output file path (default: data/deepsearchqa.json)"
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed. Install with: pip install datasets")
        return

    print("Loading DeepSearchQA dataset from HuggingFace...")
    dataset = load_dataset("google/deepsearchqa", split="eval")
    print(f"Dataset loaded: {len(dataset)} total examples")

    limit = len(dataset) if args.num_instances == -1 else min(args.num_instances, len(dataset))
    print(f"Extracting {limit} instances...")

    converted_data = []
    for i, example in enumerate(dataset):
        if i >= limit:
            break
        converted_data.append({
            "question": example["problem"],
            "answer": example["answer"] if example["answer"] else ""
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
