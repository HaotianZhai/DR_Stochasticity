"""
Modular ReAct Agent Runner

Runs the modular ReAct agent with independent temperature and seed controls
for each of the three modules:
- Summarization: Processes search results
- Reasoning: Generates inference/analysis
- Query: Issues search queries or final answers

Example usage:
    python run_multi_react_modular.py \
        --model "Qwen3-30B-A3B-Instruct-2507" \
        --output "./test_output" \
        --dataset "test_dataset.json" \
        --max_workers 10 \
        --roll_out_count 5 \
        --temperature 0 \
        --seed "1,1,2,2,3" \
        --temp-summarization 0.0 \
        --temp-reasoning 0.3 \
        --temp-query 0.0 \
        --seed-summarization 42 \
        --seed-reasoning 42 \
        --seed-query 42
"""

import argparse
import json
import os
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from tqdm import tqdm
import threading
from datetime import datetime
import time
import math

from react_agent_modular_new import ModularReactAgent


def parse_seed_list(seed_str: str, expected_count: int) -> list:
    """Parse comma-separated seed string into list of integers."""
    if not seed_str:
        return None
    try:
        seeds = [int(s.strip()) for s in seed_str.split(',')]
        if len(seeds) != expected_count:
            raise ValueError(f"Number of seeds ({len(seeds)}) must match roll_out_count ({expected_count})")
        return seeds
    except ValueError as e:
        raise ValueError(f"Invalid seed format. Please provide comma-separated integers (e.g., '1,2,3,4,5'): {e}")


def parse_float_list(value: Optional[str], expected_count: int, label: str) -> Optional[List[float]]:
    if value is None:
        return None
    try:
        temps = [float(v.strip()) for v in value.split(',')]
    except ValueError as e:
        raise ValueError(f"Invalid {label} format. Provide comma-separated floats: {e}")
    if len(temps) != expected_count:
        raise ValueError(f"{label} length ({len(temps)}) must match roll_out_count ({expected_count})")
    return temps


def main():
    parser = argparse.ArgumentParser(
        description="Run modular ReAct agent with per-module temperature/seed control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Module-specific controls:
  Each module (summarization, reasoning, query) can have independent
  temperature and seed settings. This allows studying how stochasticity
  in different components affects the final output.

Examples:
  # Basic run with default settings (temp=0 for all modules)
  python run_multi_react_modular.py --model "Qwen3-30B-A3B-Instruct-2507" \\
      --output "./output" --dataset "data.json" --roll_out_count 5

  # Higher temperature for reasoning, deterministic for others
  python run_multi_react_modular.py --model "Qwen3-30B-A3B-Instruct-2507" \\
      --output "./output" --dataset "data.json" --roll_out_count 5 \\
      --temp-reasoning 0.6 --temp-summarization 0 --temp-query 0
        """
    )
    
    # Basic arguments
    parser.add_argument("--model", type=str, default="", help="Model name/path")
    parser.add_argument("--output", type=str, default="", help="Output directory")
    parser.add_argument("--dataset", type=str, default="gaia", help="Dataset file path")
    parser.add_argument("--max_workers", type=int, default=20, help="Max parallel workers")
    parser.add_argument("--roll_out_count", type=int, default=3, help="Number of rollouts per question")
    parser.add_argument("--total_splits", type=int, default=1, help="Total number of data splits")
    parser.add_argument("--worker_split", type=int, default=1, help="Which split to process (1-indexed)")
    
    # Global temperature/seed (backward compatibility)
    parser.add_argument("--temperature", type=float, default=0.0, 
                       help="Default temperature for all modules (overridden by module-specific settings)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--presence_penalty", type=float, default=1.1, help="Presence penalty")
    parser.add_argument("--seed", type=str, default=None,
                       help="Comma-separated seeds for each rollout (e.g., '1,2,3'). Used as base seeds.")
    
    # Module-specific temperature settings
    parser.add_argument("--temp-summarization", type=float, default=None,
                       help="Temperature for summarization module (default: uses --temperature)")
    parser.add_argument("--temp-reasoning", type=float, default=None,
                       help="Temperature for reasoning module (default: uses --temperature)")
    parser.add_argument("--temp-query", type=float, default=None,
                       help="Temperature for query module (default: uses --temperature)")

    # Per-rollout temperature overrides
    parser.add_argument("--temp-summarization-list", type=str, default=None,
                       help="Comma-separated temps per rollout for summarization module (length = roll_out_count)")
    parser.add_argument("--temp-reasoning-list", type=str, default=None,
                       help="Comma-separated temps per rollout for reasoning module (length = roll_out_count)")
    parser.add_argument("--temp-query-list", type=str, default=None,
                       help="Comma-separated temps per rollout for query module (length = roll_out_count)")
    parser.add_argument("--step-temp-module", type=str, default=None,
                       choices=["summarization", "reasoning", "query"],
                       help="Apply step-level temperature override to this module")
    parser.add_argument("--step-temp-steps", type=str, default=None,
                       help="Comma-separated step numbers to override temperature (e.g., '1,2,3')")
    parser.add_argument("--step-temp-value", type=float, default=None,
                       help="Temperature to apply on specified steps (e.g., 0.5)")
    
    # Module-specific seed settings
    parser.add_argument("--seed-summarization", type=int, default=42,
                       help="Base seed for summarization module (default: 42)")
    parser.add_argument("--seed-reasoning", type=int, default=42,
                       help="Base seed for reasoning module (default: 42)")
    parser.add_argument("--seed-query", type=int, default=42,
                       help="Base seed for query module (default: 42)")

    # Mitigation toggles
    parser.add_argument("--use-ensemble", action="store_true",
                       help="Enable query ensembling (union of k=3 queries)")
    parser.add_argument("--use-consistency", action="store_true",
                       help="Enable consistency-based summarization (majority vote)")
    parser.add_argument("--use-structure", action="store_true",
                       help="Enable structured output enforcement for summarize/final report")
    
    args = parser.parse_args()
    
    # Validate splits
    if args.worker_split < 1 or args.worker_split > args.total_splits:
        print(f"Error: worker_split ({args.worker_split}) must be between 1 and total_splits ({args.total_splits})")
        exit(1)
    
    # Parse rollout seeds
    rollout_seeds = None
    if args.seed:
        try:
            rollout_seeds = parse_seed_list(args.seed, args.roll_out_count)
        except ValueError as e:
            print(f"Error: {e}")
            exit(1)

    try:
        temp_summarization_list = parse_float_list(args.temp_summarization_list, args.roll_out_count, "temp-summarization-list")
        temp_reasoning_list = parse_float_list(args.temp_reasoning_list, args.roll_out_count, "temp-reasoning-list")
        temp_query_list = parse_float_list(args.temp_query_list, args.roll_out_count, "temp-query-list")
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    
    # Build module configurations
    default_temp = args.temperature
    module_configs = {
        "summarization": {
            "temperature": args.temp_summarization if args.temp_summarization is not None else default_temp,
            "seed": args.seed_summarization
        },
        "reasoning": {
            "temperature": args.temp_reasoning if args.temp_reasoning is not None else default_temp,
            "seed": args.seed_reasoning
        },
        "query": {
            "temperature": args.temp_query if args.temp_query is not None else default_temp,
            "seed": args.seed_query
        }
    }
    
    # Setup output directories
    model_name = os.path.basename(args.model.rstrip('/'))
    model_dir = os.path.join(args.output, f"{model_name}_modular")
    dataset_name = os.path.basename(args.dataset.rstrip('/'))
    dataset_dir = os.path.join(model_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Print configuration
    print("=" * 70)
    print("MODULAR REACT AGENT CONFIGURATION")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {dataset_dir}")
    print(f"Rollouts: {args.roll_out_count}")
    print(f"Workers: {args.max_workers}")
    print(f"Data split: {args.worker_split}/{args.total_splits}")
    print()
    print("Module Settings:")
    print(f"  Summarization: temp={module_configs['summarization']['temperature']}, seed={module_configs['summarization']['seed']}")
    print(f"  Reasoning:     temp={module_configs['reasoning']['temperature']}, seed={module_configs['reasoning']['seed']}")
    print(f"  Query:         temp={module_configs['query']['temperature']}, seed={module_configs['query']['seed']}")
    print("Mitigations:")
    print(f"  use_ensemble:   {args.use_ensemble}")
    print(f"  use_consistency:{args.use_consistency}")
    print(f"  use_structure:  {args.use_structure}")
    if rollout_seeds:
        print(f"  Rollout seeds: {rollout_seeds}")
    print("=" * 70)
    
    # Load dataset
    data_filepath = args.dataset
    try:
        if data_filepath.endswith(".json"):
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = json.load(f)
            if not isinstance(items, list):
                raise ValueError("Input JSON must be a list of objects.")
        elif data_filepath.endswith(".jsonl"):
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f]
        else:
            raise ValueError("Unsupported file extension. Please use .json or .jsonl files.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {data_filepath}")
        exit(1)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error reading or parsing input file {data_filepath}: {e}")
        exit(1)
    
    # Apply data splitting
    total_items = len(items)
    items_per_split = math.ceil(total_items / args.total_splits)
    start_idx = (args.worker_split - 1) * items_per_split
    end_idx = min(args.worker_split * items_per_split, total_items)
    items = items[start_idx:end_idx]
    
    print(f"Total items: {total_items}, Processing: {start_idx}-{end_idx-1} ({len(items)} items)")
    
    # Setup output files
    if args.total_splits > 1:
        output_files = {i: os.path.join(dataset_dir, f"iter{i}_split{args.worker_split}of{args.total_splits}.jsonl") 
                       for i in range(1, args.roll_out_count + 1)}
    else:
        output_files = {i: os.path.join(dataset_dir, f"iter{i}.jsonl") 
                       for i in range(1, args.roll_out_count + 1)}
    
    # Track processed queries
    processed_queries_per_rollout = {}
    for rollout_idx in range(1, args.roll_out_count + 1):
        output_file = output_files[rollout_idx]
        processed_queries = set()
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if "question" in data and "error" not in data:
                                processed_queries.add(data["question"].strip())
                        except json.JSONDecodeError:
                            pass
            except FileNotFoundError:
                pass
        processed_queries_per_rollout[rollout_idx] = processed_queries
    
    # Build task list
    tasks_to_run = []
    per_rollout_counts = {i: 0 for i in range(1, args.roll_out_count + 1)}
    
    for rollout_idx in range(1, args.roll_out_count + 1):
        processed = processed_queries_per_rollout[rollout_idx]
        for item in items:
            question = item.get("question", "").strip()
            if not question:
                try:
                    user_msg = item["messages"][1]["content"]
                    question = user_msg.split("User:")[1].strip() if "User:" in user_msg else user_msg
                    item["question"] = question
                except:
                    continue
            
            if question and question not in processed:
                task = {
                    "item": item.copy(),
                    "rollout_idx": rollout_idx,
                    "planning_port": 6001,
                }
                
                # Set module seeds from rollout seed (same seed for all modules in a run)
                if rollout_seeds:
                    run_seed = rollout_seeds[rollout_idx - 1]
                    task["module_seeds"] = {
                        "summarization": run_seed,
                        "reasoning": run_seed,
                        "query": run_seed,
                    }

                module_temps = {}
                if temp_summarization_list is not None:
                    module_temps["summarization"] = temp_summarization_list[rollout_idx - 1]
                if temp_reasoning_list is not None:
                    module_temps["reasoning"] = temp_reasoning_list[rollout_idx - 1]
                if temp_query_list is not None:
                    module_temps["query"] = temp_query_list[rollout_idx - 1]
                if module_temps:
                    task["module_temps"] = module_temps
                
                tasks_to_run.append(task)
                per_rollout_counts[rollout_idx] += 1
    
    # Print task summary
    print(f"\nQuestions in split: {len(items)}")
    for rollout_idx in range(1, args.roll_out_count + 1):
        print(f"Rollout {rollout_idx}: processed={len(processed_queries_per_rollout[rollout_idx])}, pending={per_rollout_counts[rollout_idx]}")
    
    if not tasks_to_run:
        print("\nAll rollouts completed. No tasks to run.")
        return
    
    # Initialize agent
    llm_cfg = {
        'model': args.model,
        'generate_cfg': {
            'max_input_tokens': 320000,
            'max_retries': 10,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'presence_penalty': args.presence_penalty
        },
        'model_type': 'qwen_dashscope'
    }
    
    step_temp_config = {}
    if args.step_temp_module and args.step_temp_steps and args.step_temp_value is not None:
        try:
            steps = {int(s.strip()) for s in args.step_temp_steps.split(',') if s.strip()}
        except ValueError:
            print("Error: --step-temp-steps must be comma-separated integers (e.g., '1,2,3')")
            exit(1)
        step_temp_config[args.step_temp_module] = {
            "steps": steps,
            "temp": float(args.step_temp_value)
        }

    agent = ModularReactAgent(
        llm=llm_cfg,
        function_list=["search"],
        module_configs=module_configs,
        step_temp_config=step_temp_config,
        mitigation_config={
            "use_ensemble": args.use_ensemble,
            "use_consistency": args.use_consistency,
            "use_structure": args.use_structure,
        }
    )
    
    # Run tasks
    write_locks = {i: threading.Lock() for i in range(1, args.roll_out_count + 1)}
    
    print(f"\nStarting {len(tasks_to_run)} tasks with {args.max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {
            executor.submit(agent._run, task, args.model): task
            for task in tasks_to_run
        }
        
        for future in tqdm(as_completed(future_to_task), total=len(tasks_to_run), desc="Processing"):
            task_info = future_to_task[future]
            rollout_idx = task_info["rollout_idx"]
            output_file = output_files[rollout_idx]
            
            try:
                result = future.result()
                # Add effective module config to result (with seeds applied)
                effective_config = {
                    module: {
                        "temperature": module_configs[module]["temperature"],
                        "seed": task_info["module_seeds"][module] if "module_seeds" in task_info else module_configs[module]["seed"]
                    }
                    for module in ["summarization", "reasoning", "query"]
                }
                result["module_configs"] = effective_config
                result["rollout_idx"] = rollout_idx
                
                with write_locks[rollout_idx]:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            except concurrent.futures.TimeoutError:
                question = task_info["item"].get("question", "")
                print(f'Timeout: "{question[:50]}..." (Rollout {rollout_idx})')
                error_result = {
                    "question": question,
                    "answer": task_info["item"].get("answer", ""),
                    "rollout_idx": rollout_idx,
                    "error": "Timeout",
                    "messages": [],
                    "prediction": "[Failed]"
                }
                with write_locks[rollout_idx]:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
            
            except Exception as exc:
                question = task_info["item"].get("question", "")
                print(f'Error for "{question[:50]}..." (Rollout {rollout_idx}): {exc}')
                error_result = {
                    "question": question,
                    "answer": task_info["item"].get("answer", ""),
                    "rollout_idx": rollout_idx,
                    "error": str(exc),
                    "messages": [],
                    "prediction": "[Failed]"
                }
                with write_locks[rollout_idx]:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
    
    print(f"\nAll {args.roll_out_count} rollouts completed!")
    print(f"Results saved to: {dataset_dir}")


if __name__ == "__main__":
    main()
