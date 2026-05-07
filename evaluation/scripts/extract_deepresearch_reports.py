#!/usr/bin/env python3
"""
Script to extract reports from DeepResearch JSONL output format for stochasticity analysis.

This script processes JSONL files from the modular ReAct agent output and:
1. Extracts the research question and final prediction/report from each entry
2. Converts them to trajectory JSON files compatible with the stochasticity pipeline
3. Handles the different format where reports appear in the "prediction" field

Usage:
    python extract_deepresearch_reports.py --input-dir /path/to/jsonl/dir --output-dir /path/to/output
"""

import json
import os
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
from datetime import datetime


def generate_session_id(question: str, iteration: str = "1") -> str:
    question_hash = hashlib.md5(question.encode()).hexdigest()[:8]
    return f"deepresearch_iter{iteration}_{question_hash}"


def extract_question_id(question: str) -> str:
    return hashlib.md5(question.encode()).hexdigest()[:12]


def parse_deepresearch_jsonl(file_path: Path, iteration: str = "1") -> List[Dict[str, Any]]:
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entry['_line_num'] = line_num
                entry['_iteration'] = iteration
                entry['_source_file'] = str(file_path)
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"  Warning: Failed to parse line {line_num} in {file_path.name}: {e}")
    return entries


def convert_to_trajectory_format(entry: Dict[str, Any], iteration: str) -> Dict[str, Any]:
    question = entry.get("question", "")
    prediction = entry.get("prediction", "")
    ground_truth = entry.get("answer", "")
    messages = entry.get("messages", [])
    termination = entry.get("termination", "unknown")

    session_id = generate_session_id(question, iteration)
    question_id = extract_question_id(question)

    trajectory_steps = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        step = {
            "step_number": i,
            "input_state": {
                "messages": [{
                    "type": "human" if role == "user" else ("ai" if role == "assistant" else "system"),
                    "content": content
                }]
            },
            "output_state": {}
        }
        trajectory_steps.append(step)

    return {
        "session_id": session_id,
        "question_id": question_id,
        "iteration": iteration,
        "research_question": question,
        "ground_truth_answer": ground_truth,
        "final_report": prediction,
        "termination_reason": termination,
        "trajectory_steps": trajectory_steps,
        "metadata": {
            "source_format": "deepresearch_jsonl",
            "source_file": entry.get("_source_file", ""),
            "line_number": entry.get("_line_num", 0),
            "iteration": iteration,
            "num_messages": len(messages)
        }
    }


def extract_iteration_from_filename(filename: str) -> str:
    match = re.search(r'iter(\d+)', filename, re.IGNORECASE)
    if match:
        return match.group(1)
    return "1"


def process_deepresearch_files(
    input_paths: List[Path],
    output_dir: Path,
    verbose: bool = True
) -> Tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    total_processed = 0
    total_errors = 0

    jsonl_files = []
    for path in input_paths:
        if path.is_file() and path.suffix == '.jsonl':
            jsonl_files.append(path)
        elif path.is_dir():
            jsonl_files.extend(path.glob("*.jsonl"))

    if not jsonl_files:
        print("No JSONL files found to process.")
        return 0, 0

    print(f"\nFound {len(jsonl_files)} JSONL file(s) to process")
    question_to_sessions = {}

    for jsonl_file in sorted(jsonl_files):
        print(f"\nProcessing: {jsonl_file.name}")
        iteration = extract_iteration_from_filename(jsonl_file.name)
        entries = parse_deepresearch_jsonl(jsonl_file, iteration)
        print(f"  Found {len(entries)} entries (iteration={iteration})")

        for entry in entries:
            try:
                trajectory = convert_to_trajectory_format(entry, iteration)
                session_id = trajectory["session_id"]
                question_id = trajectory["question_id"]
                question = trajectory["research_question"]

                if question_id not in question_to_sessions:
                    question_to_sessions[question_id] = {
                        "question": question, "sessions": []
                    }
                question_to_sessions[question_id]["sessions"].append({
                    "session_id": session_id,
                    "iteration": iteration,
                    "file": str(output_dir / f"deep_research_trajectory_{session_id}.json")
                })

                output_file = output_dir / f"deep_research_trajectory_{session_id}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(trajectory, f, indent=2, ensure_ascii=False)

                if verbose:
                    q_preview = question[:50] + "..." if len(question) > 50 else question
                    print(f"  Saved: {output_file.name} ({q_preview})")
                total_processed += 1
            except Exception as e:
                print(f"  Error processing entry: {e}")
                total_errors += 1

    index_file = output_dir / "question_groups_index.json"
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_questions": len(question_to_sessions),
            "total_trajectories": total_processed,
            "question_groups": question_to_sessions
        }, f, indent=2, ensure_ascii=False)

    return total_processed, total_errors


def main():
    parser = argparse.ArgumentParser(
        description="Extract reports from DeepResearch JSONL files for stochasticity analysis"
    )
    parser.add_argument("--input-dir", type=str, help="Directory containing JSONL files")
    parser.add_argument("--input-files", type=str, nargs="+", help="Specific JSONL files to process")
    parser.add_argument("--output-dir", type=str, default="./trajectories",
                       help="Directory to save converted trajectory files")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    args = parser.parse_args()

    input_paths = []
    if args.input_files:
        input_paths.extend([Path(f) for f in args.input_files])
    if args.input_dir:
        input_paths.append(Path(args.input_dir))
    if not input_paths:
        print("Error: Please specify --input-dir or --input-files")
        return 1

    output_dir = Path(args.output_dir)
    print("=" * 80)
    print("DEEPRESEARCH JSONL TO TRAJECTORY CONVERTER")
    print("=" * 80)

    total_processed, total_errors = process_deepresearch_files(
        input_paths=input_paths,
        output_dir=output_dir,
        verbose=not args.quiet
    )

    print(f"\nTotal trajectories created: {total_processed}")
    print(f"Total errors: {total_errors}")
    return 0


if __name__ == "__main__":
    exit(main())
