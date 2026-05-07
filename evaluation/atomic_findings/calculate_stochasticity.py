#!/usr/bin/env python3
"""
Calculate Stochasticity Metrics for Deep Research Systems

This script computes three metrics to evaluate stochasticity:
1. Answer-level pairwise stochasticity (with LLM semantic comparison)
2. Finding-level pairwise stochasticity (normalized cosine distance)
3. Citation-level stochasticity (set-level citation overlap)

Key features:
- Groups trajectories by question ID (e.g., _0_*, _1_* as separate batches)
- Uses LLM to semantically compare answers instead of exact string matching
- Provides per-group and aggregated metrics
- Uses local OpenAI-compatible API (Qwen/Qwen3-30B-A3B-Instruct-2507)

Usage:
    python calculate_stochasticity.py <clustered_findings.json> <answers_dir> [OPTIONS]
    
Example:
    python calculate_stochasticity.py \
        output/test_results.json \
        answers/ \
        --llm-base-url http://localhost:30000/v1 \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507
"""

import json
import argparse
import os
import re
import time
import math
from pathlib import Path
from typing import List, Dict, Set, Tuple, Any, Optional, Union
from itertools import combinations
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from openai import OpenAI
import httpx
from urllib.parse import urlparse, parse_qsl, urlencode

# Default configuration for local LLM server
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:30000/v1")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
DEFAULT_LLM_API_KEY = os.getenv("TOGETHER_API_KEY", "EMPTY")


def create_openai_client(base_url: str, api_key: str = None) -> OpenAI:
    key = api_key or DEFAULT_LLM_API_KEY
    try:
        return OpenAI(api_key=key, base_url=base_url)
    except (PermissionError, OSError) as exc:
        print(f"Warning: OpenAI client SSL setup failed ({exc}); retrying with verify=False.")
        return OpenAI(
            api_key=key,
            base_url=base_url,
            http_client=httpx.Client(verify=False)
        )


def call_llm_for_answer_comparison(
    answer1: str,
    answer2: str,
    llm_client: Optional[OpenAI] = None,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    model: str = DEFAULT_LLM_MODEL,
    max_retries: int = 3
) -> bool:
    """
    Use LLM to determine if two answers are semantically equivalent.
    Uses a local OpenAI-compatible API.
    
    Args:
        answer1: First answer text
        answer2: Second answer text
        llm_client: Optional pre-initialized OpenAI client
        llm_base_url: Base URL for local LLM server
        model: Model name
        max_retries: Maximum number of retry attempts
        
    Returns:
        True if answers are semantically equivalent, False otherwise
    """
    prompt = f"""Compare these two answers to determine if they are semantically equivalent.
Two answers are semantically equivalent if they convey the same core information, even if worded differently.

Answer 1: {answer1}

Answer 2: {answer2}

Are these answers semantically equivalent? Respond with only "YES" or "NO".

Response:"""

    # Initialize client if not provided
    if llm_client is None:
        llm_client = create_openai_client(llm_base_url)
    
    for attempt in range(max_retries):
        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
                seed=42  # Required by SGLang when temperature=0
            )
            
            result = response.choices[0].message.content.strip().upper()
            return "YES" in result
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))  # Exponential backoff
            else:
                print(f"Warning: LLM call failed after {max_retries} retries: {e}. Falling back to exact match.")
                return answer1.strip().lower() == answer2.strip().lower()
    
    return answer1.strip().lower() == answer2.strip().lower()


def extract_question_id(filename: str) -> Optional[str]:
    """
    Extract question ID from filename.
    
    Supports multiple formats:
    1. DeepResearch format: 'qa_answer_deepresearch_iter1_800888c0.json' -> '800888c0'
    2. Alternative format: 'qa_answer_0_1.json' -> '0'
        
    Returns:
        Question ID as string, or None if pattern doesn't match
    """
    # Format 1: DeepResearch format with iter and hex hash
    # Pattern: *_iter{N}_{hexhash}
    match = re.search(r'_iter\d+_([a-f0-9]+)', filename)
    if match:
        return match.group(1)  # Hex hash is question ID
    
    # Format 2: Legacy format with _X_Y where X is question ID, Y is run ID
    match = re.search(r'_(\d+)_(\d+)', filename)
    if match:
        return match.group(1)  # First number is question ID
    
    return None


def load_clustered_findings(findings_path: str) -> Dict[str, Any]:
    """Load the clustered findings JSON file."""
    with open(findings_path, 'r') as f:
        return json.load(f)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def _extract_json_array(text: str) -> Optional[List[str]]:
    text = _strip_code_fences(text)
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
        if isinstance(data, dict) and "urls" in data and isinstance(data["urls"], list):
            return [str(item).strip() for item in data["urls"] if str(item).strip()]
    except Exception:
        return None
    return None


def _extract_urls_regex(text: str) -> List[str]:
    urls = re.findall(r"https?://[^\s\)\]\}<>\"']+", text)
    return [url.strip().rstrip(").,;") for url in urls]


def _extract_sources_section(report_text: str) -> str:
    match = re.search(r"^\s*##\s+Sources\b", report_text, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        return report_text[match.start():]
    return report_text


def extract_urls_from_report_llm(
    report_text: str,
    llm_client: Optional[OpenAI] = None,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    model: str = DEFAULT_LLM_MODEL,
    max_retries: int = 3
) -> List[str]:
    """
    Use LLM to extract URLs from a final report. Returns a list of URLs.
    Falls back to regex extraction on error.
    """
    if not report_text or not report_text.strip():
        return []
    report_section = _extract_sources_section(report_text)
    prompt = (
        "Extract all URLs from the report text below. "
        "Return ONLY a JSON array of URL strings. "
        "If no URLs are present, return an empty JSON array [].\n\n"
        f"REPORT:\n{report_section}"
    )
    if llm_client is None:
        llm_client = create_openai_client(llm_base_url)
    for attempt in range(max_retries):
        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
                seed=42
            )
            content = response.choices[0].message.content or ""
            urls = _extract_json_array(content)
            if urls is not None:
                return urls
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
            else:
                print(f"Warning: URL extraction failed after {max_retries} retries: {e}")
                break
    return _extract_urls_regex(report_text)


def normalize_url(url: str) -> str:
    """Normalize URL to improve clustering consistency."""
    if not url:
        return ""
    url = url.strip().strip("<>\"'()[]{}.,;")
    parsed = urlparse(url)
    if not parsed.scheme and not parsed.netloc:
        parsed = urlparse(f"http://{url}")
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = re.sub(r"/+$", "", parsed.path or "")
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    filtered_pairs = []
    for key, value in query_pairs:
        lowered = key.lower()
        if lowered.startswith("utm_") or lowered in {"fbclid", "gclid", "igshid", "ref"}:
            continue
        filtered_pairs.append((key, value))
    query = urlencode(filtered_pairs, doseq=True)
    normalized = f"{host}{path}"
    if query:
        normalized = f"{normalized}?{query}"
    return normalized


def resolve_trajectory_path(answer_source_path: Path, trajectory_file: str) -> Optional[Path]:
    if not trajectory_file:
        return None
    trajectory_path = Path(trajectory_file)
    if trajectory_path.is_absolute():
        return trajectory_path if trajectory_path.exists() else None
    for base in [answer_source_path.parent, *answer_source_path.parents, Path.cwd()]:
        candidate = base / trajectory_file
        if candidate.exists():
            return candidate
    return None


def extract_final_report_from_trajectory(trajectory_data: Dict[str, Any]) -> str:
    if isinstance(trajectory_data.get('final_report'), str):
        return trajectory_data.get('final_report', '')
    steps = trajectory_data.get('trajectory_steps', []) or []
    for step in reversed(steps):
        for key in ("final_report", "report", "final_answer", "final_response"):
            if isinstance(step.get(key), str):
                return step.get(key, '')
            output_state = step.get("output_state", {}) or {}
            if isinstance(output_state.get(key), str):
                return output_state.get(key, '')
    return ""


def load_reference_qa_pairs(reference_path: str) -> Dict[str, str]:
    """
    Load reference QA pairs from a JSON file.
    
    Args:
        reference_path: Path to the reference JSON file
        
    Returns:
        Dict mapping question text -> ground truth answer
    """
    with open(reference_path, 'r') as f:
        data = json.load(f)
    
    # Handle both list format and dict format
    qa_map = {}
    if isinstance(data, list):
        for item in data:
            question = item.get('question', '')
            answer = item.get('answer', '')
            if question:
                qa_map[question] = answer
    elif isinstance(data, dict):
        # Might be a dict with question as key
        for k, v in data.items():
            if isinstance(v, dict) and 'answer' in v:
                qa_map[k] = v['answer']
            elif isinstance(v, str):
                qa_map[k] = v
    
    return qa_map


def call_llm_for_accuracy_check(
    generated_answer: str,
    ground_truth: str,
    question: str,
    llm_client: Optional[OpenAI] = None,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    model: str = DEFAULT_LLM_MODEL,
    max_retries: int = 3
) -> Tuple[bool, str]:
    """
    Use LLM to determine if a generated answer is correct compared to ground truth.
    Uses a local OpenAI-compatible API.
    
    Args:
        generated_answer: The answer generated by the model
        ground_truth: The ground truth answer
        question: The original question (for context)
        llm_client: Optional pre-initialized OpenAI client
        llm_base_url: Base URL for local LLM server
        model: Model name
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (is_correct: bool, explanation: str)
    """
    prompt = f"""You are a strict evaluator comparing a generated answer against the ground truth answer for a question.

Question: {question}

Ground Truth Answer: "{ground_truth}"

Generated Answer: "{generated_answer}"

EVALUATION CRITERIA:
1. The generated answer must contain the SAME core factual answer as the ground truth
2. Format variations are acceptable: "1988-96" = "1988 to 1996" = "1988-1996" (all equivalent)
3. If the ground truth provides a specific answer (like a name, date, or fact), the generated answer must contain that SAME information
4. If the generated answer says "not found", "unknown", or similar but the ground truth has a specific answer, mark as INCORRECT
5. Extra context or elaboration in the generated answer is fine as long as the core answer matches

Examples:
- Ground truth: "1988-96", Generated: "1988 to 1996" → CORRECT (same date range)
- Ground truth: "Ireland v Romania", Generated: "No match found" → INCORRECT (missing the actual answer)
- Ground truth: "John Smith", Generated: "The person is John Smith who was born in 1950" → CORRECT (contains the name)

Respond with ONLY "CORRECT" or "INCORRECT" followed by a brief explanation.

Response:"""

    # Initialize client if not provided
    if llm_client is None:
        llm_client = create_openai_client(llm_base_url)
    
    for attempt in range(max_retries):
        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
                seed=42  # Required by SGLang when temperature=0
            )
            result = response.choices[0].message.content.strip()
            is_correct = result.upper().startswith("CORRECT")
            return is_correct, result
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))  # Exponential backoff
            else:
                print(f"Warning: LLM accuracy check failed after {max_retries} retries: {e}. Falling back to string matching.")
                is_correct = ground_truth.lower().strip() in generated_answer.lower().strip()
                return is_correct, f"Fallback due to error: {e}"
    
    is_correct = ground_truth.lower().strip() in generated_answer.lower().strip()
    return is_correct, "Fallback after retries"


def calculate_accuracy(
    grouped_answers: Dict[str, List[Dict[str, Any]]],
    reference_qa: Dict[str, str],
    llm_client: Optional[OpenAI] = None,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
    use_llm: bool = True,
    max_workers: int = 4,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Calculate accuracy by comparing generated answers with ground truth.
    
    Args:
        grouped_answers: Dict mapping question_id -> list of answer dicts
        reference_qa: Dict mapping question text -> ground truth answer
        llm_client: Optional pre-initialized OpenAI client
        llm_base_url: Base URL for local LLM server
        llm_model: Model name
        use_llm: Whether to use LLM for comparison
        max_workers: Number of parallel workers
        verbose: Whether to print progress
        
    Returns:
        Dict with accuracy metrics per group and aggregate
    """
    if verbose:
        print(f"\n{'='*70}")
        print("CALCULATING ACCURACY")
        print(f"{'='*70}")
        print(f"  Reference QA pairs: {len(reference_qa)}")
    
    # Initialize client if not provided
    if llm_client is None and use_llm:
        llm_client = create_openai_client(llm_base_url)
    
    per_group_accuracy = []
    all_correct = 0
    all_total = 0
    
    for group_id, answers in grouped_answers.items():
        group_correct = 0
        group_total = len(answers)
        group_details = []
        
        for answer_data in answers:
            # Extract question and generated answer
            question = answer_data.get('research_question', '')
            
            # Handle different answer formats
            if 'answer_data' in answer_data:
                generated_answer = answer_data['answer_data'].get('answer', '')
            elif 'answer' in answer_data:
                generated_answer = answer_data['answer']
            else:
                generated_answer = ''
            
            # Find ground truth from reference
            ground_truth = reference_qa.get(question, None)
            
            if ground_truth is None:
                # Try fuzzy matching if exact match fails
                for ref_q, ref_a in reference_qa.items():
                    if question.strip()[:100] == ref_q.strip()[:100]:  # Match first 100 chars
                        ground_truth = ref_a
                        break
            
            if ground_truth is None:
                if verbose:
                    print(f"  ⚠️  Warning: No ground truth found for question: {question[:50]}...")
                continue
            
            # Compare using LLM
            if use_llm:
                is_correct, explanation = call_llm_for_accuracy_check(
                    generated_answer=generated_answer,
                    ground_truth=ground_truth,
                    question=question,
                    llm_client=llm_client,
                    llm_base_url=llm_base_url,
                    model=llm_model
                )
            else:
                # Simple string matching
                is_correct = ground_truth.lower().strip() in generated_answer.lower().strip()
                explanation = "String matching"
            
            if is_correct:
                group_correct += 1
            
            group_details.append({
                'question': question[:100] + '...' if len(question) > 100 else question,
                'ground_truth': ground_truth,
                'generated_answer': generated_answer[:200] + '...' if len(generated_answer) > 200 else generated_answer,
                'is_correct': is_correct,
                'explanation': explanation[:100] if len(explanation) > 100 else explanation
            })
        
        group_accuracy = group_correct / group_total if group_total > 0 else 0.0
        all_correct += group_correct
        all_total += group_total
        
        per_group_accuracy.append({
            'group_id': group_id,
            'correct': group_correct,
            'total': group_total,
            'accuracy': group_accuracy,
            'details': group_details
        })
        
        if verbose:
            print(f"\n  Group '{group_id}': {group_correct}/{group_total} correct ({group_accuracy:.2%})")
    
    overall_accuracy = all_correct / all_total if all_total > 0 else 0.0
    
    if verbose:
        print(f"\n  OVERALL ACCURACY: {all_correct}/{all_total} ({overall_accuracy:.2%})")
    
    return {
        'overall_accuracy': overall_accuracy,
        'total_correct': all_correct,
        'total_questions': all_total,
        'per_group_accuracy': per_group_accuracy
    }


def load_answers_grouped(answers_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load answer JSON files grouped by question ID.
    
    Returns:
        Dict mapping question_id -> list of answer dicts (with source_file added)
    """
    answers_path = Path(answers_dir)
    grouped_answers = defaultdict(list)
    
    if answers_path.is_file():
        # Single file - extract question ID from filename
        with open(answers_path, 'r') as f:
            data = json.load(f)
            data['source_file'] = answers_path.name  # Add source filename
            data['source_path'] = str(answers_path.resolve())
            question_id = extract_question_id(answers_path.name)
            if question_id:
                grouped_answers[question_id].append(data)
            else:
                grouped_answers['default'].append(data)
    else:
        # Directory - group all JSON files by question ID
        for json_file in sorted(answers_path.glob('*.json')):
            with open(json_file, 'r') as f:
                data = json.load(f)
                data['source_file'] = json_file.name  # Add source filename
                data['source_path'] = str(json_file.resolve())
                question_id = extract_question_id(json_file.name)
                if question_id:
                    grouped_answers[question_id].append(data)
                else:
                    grouped_answers['default'].append(data)
    
    return dict(grouped_answers)


def group_canonical_sets_by_answers(
    canonical_sets: List[Set[int]],
    grouped_answers: Dict[str, List[Dict[str, Any]]],
    source_filenames: Optional[List[str]] = None
) -> Dict[str, List[Set[int]]]:
    """
    Group canonical finding sets to match answer groups.
    
    If source_filenames are provided, uses explicit filename matching.
    Otherwise, falls back to position-based matching (legacy behavior).
    
    Args:
        canonical_sets: List of all canonical finding sets
        grouped_answers: Dict mapping question_id -> list of answers
        source_filenames: Optional list of source filenames for explicit matching
        
    Returns:
        Dict mapping question_id -> list of canonical finding sets
    """
    if source_filenames and len(source_filenames) == len(canonical_sets):
        # Use explicit filename matching
        return group_canonical_sets_by_filename(
            canonical_sets, grouped_answers, source_filenames
        )
    else:
        # Fallback to legacy position-based matching
        if source_filenames:
            print("Warning: source_filenames length mismatch, falling back to position-based matching")
        return group_canonical_sets_by_position(canonical_sets, grouped_answers)


def group_canonical_sets_by_filename(
    canonical_sets: List[Set[int]],
    grouped_answers: Dict[str, List[Dict[str, Any]]],
    source_filenames: List[str]
) -> Dict[str, List[Set[int]]]:
    """
    Group canonical finding sets using explicit filename matching.
    
    Extracts question_id and run_id from filenames and matches them.
    
    Args:
        canonical_sets: List of all canonical finding sets
        grouped_answers: Dict mapping question_id -> list of answers
        source_filenames: List of source filenames (e.g., ['atomic_facts_0_1.json', ...])
        
    Returns:
        Dict mapping question_id -> list of canonical finding sets
    """
    # Build a map: (question_id, run_id) -> canonical_set
    filename_to_set = {}
    for idx, filename in enumerate(source_filenames):
        question_id = extract_question_id(filename)
        run_id = extract_run_id(filename)
        if question_id and run_id:
            filename_to_set[(question_id, run_id)] = canonical_sets[idx]
    
    # Build a map: (question_id, run_id) -> answer for grouped_answers
    answer_keys = {}
    for qid, answers in grouped_answers.items():
        for answer in answers:
            # Extract run_id from answer source_file or filename
            answer_filename = answer.get('source_file', answer.get('filename', ''))
            if answer_filename:
                run_id = extract_run_id(answer_filename)
                if run_id:
                    answer_keys[(qid, run_id)] = answer
    
    # Match canonical sets to answer groups
    grouped_sets = {}
    unmatched_canonical = []
    unmatched_answers = []
    
    for question_id in sorted(grouped_answers.keys()):
        grouped_sets[question_id] = []
        
        for answer in grouped_answers[question_id]:
            answer_filename = answer.get('source_file', answer.get('filename', ''))
            run_id = extract_run_id(answer_filename)
            
            if run_id and (question_id, run_id) in filename_to_set:
                grouped_sets[question_id].append(filename_to_set[(question_id, run_id)])
            else:
                unmatched_answers.append(f"{question_id}_{run_id}" if run_id else answer_filename)
    
    # Check for unmatched canonical sets
    matched_keys = set()
    for qid in grouped_sets:
        for answer in grouped_answers[qid]:
            answer_filename = answer.get('source_file', answer.get('filename', ''))
            run_id = extract_run_id(answer_filename)
            if run_id:
                matched_keys.add((qid, run_id))
    
    for key in filename_to_set:
        if key not in matched_keys:
            unmatched_canonical.append(f"{key[0]}_{key[1]}")
    
    # Report mismatches
    if unmatched_answers or unmatched_canonical:
        print("\n⚠️  Warning: Filename matching found mismatches:")
        if unmatched_answers:
            print(f"  Answers without matching canonical sets: {unmatched_answers}")
        if unmatched_canonical:
            print(f"  Canonical sets without matching answers: {unmatched_canonical}")
    
    return grouped_sets


def group_canonical_sets_by_position(
    canonical_sets: List[Set[int]],
    grouped_answers: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Set[int]]]:
    """
    Group canonical finding sets using position-based matching (legacy).
    
    WARNING: This assumes files were processed in the same order, which is fragile.
    
    Args:
        canonical_sets: List of all canonical finding sets
        grouped_answers: Dict mapping question_id -> list of answers
        
    Returns:
        Dict mapping question_id -> list of canonical finding sets
    """
    grouped_sets = {}
    idx = 0
    
    for question_id in sorted(grouped_answers.keys()):
        num_answers = len(grouped_answers[question_id])
        # Take the next N canonical sets for this question
        grouped_sets[question_id] = canonical_sets[idx:idx + num_answers]
        idx += num_answers
    
    return grouped_sets


def extract_run_id(filename: str) -> Optional[str]:
    """
    Extract run ID from filename.
    
    Supports multiple formats:
    1. DeepResearch format: 'qa_answer_deepresearch_iter1_800888c0.json' -> '1'
    2. Alternative format: 'qa_answer_0_1.json' -> '1'
        
    Returns:
        Run ID as string, or None if pattern doesn't match
    """
    # Format 1: DeepResearch format with iter{N}
    # Pattern: *_iter{N}_*
    match = re.search(r'_iter(\d+)_', filename)
    if match:
        return match.group(1)  # Iteration number is run ID
    
    # Format 2: Legacy format with _X_Y where X is question ID, Y is run ID
    match = re.search(r'_(\d+)_(\d+)', filename)
    if match:
        return match.group(2)  # Second number is run ID
    
    return None


def extract_canonical_finding_sets(clustered_data: Dict[str, Any]) -> List[Set[int]]:
    """
    Extract canonical finding sets for each report.
    
    Returns:
        List of sets, where each set contains the cluster_ids (canonical findings)
        for that report.
    """
    canonical_files = clustered_data['canonical_files']
    
    # canonical_files already contains cluster_ids (not finding_indices)
    # Just convert each list to a set
    canonical_sets = [set(cluster_ids) for cluster_ids in canonical_files]
    
    return canonical_sets


def normalized_cosine_distance(set_a: Set, set_b: Set) -> float:
    """
    Compute normalized cosine distance between two binary sets.
    
    For binary vectors x, y with L2 normalization:
      d(x, y) = 1 - (x · y)
             = 1 - |A ∩ B| / sqrt(|A||B|)
    
    Edge cases:
    - If both sets are empty, return 0.0 (no distance).
    - If only one set is empty, return 1.0 (max distance).
    """
    size_a = len(set_a)
    size_b = len(set_b)
    if size_a == 0 and size_b == 0:
        return 0.0
    if size_a == 0 or size_b == 0:
        return 1.0
    intersection = len(set_a & set_b)
    denom = math.sqrt(size_a * size_b)
    similarity = intersection / denom if denom > 0 else 0.0
    return 1.0 - similarity


def average_pairwise_distance(sets: List[Set]) -> float:
    """Average normalized cosine distance across all unordered pairs."""
    if len(sets) < 2:
        return 0.0
    total_distance = 0.0
    num_pairs = 0
    for i, j in combinations(range(len(sets)), 2):
        total_distance += normalized_cosine_distance(sets[i], sets[j])
        num_pairs += 1
    return total_distance / num_pairs if num_pairs else 0.0


def support_size_tv(sets: List[Set]) -> float:
    """
    TV of the support size ||X||_0 using pairwise estimator.
    For unordered pairs: (1 / (n(n-1))) * sum_{i<j} (|Si| - |Sj|)^2
    """
    n = len(sets)
    if n < 2:
        return 0.0
    sizes = [len(s) for s in sets]
    total = 0.0
    for i, j in combinations(range(n), 2):
        diff = sizes[i] - sizes[j]
        total += diff * diff
    return total / (n * (n - 1))


def metric_1_answer_pairwise_llm(
    answers: List[Dict[str, Any]],
    llm_client: Optional[OpenAI] = None,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
    use_llm: bool = True,
    max_workers: int = 4
) -> float:
    """
    METRIC 1: Answer-level pairwise stochasticity (with LLM semantic comparison)
    
    Uses LLM to determine if answers are semantically equivalent.
    Returns the average over all pairs.
    """
    if len(answers) < 2:
        return 0.0
    
    # Initialize client if not provided
    if llm_client is None and use_llm:
        llm_client = create_openai_client(llm_base_url)
    
    # Extract the actual answer strings
    answer_texts = []
    for ans in answers:
        # Handle different possible structures
        if 'answer_data' in ans:
            answer_texts.append(ans['answer_data'].get('answer', ''))
        elif 'answer' in ans:
            answer_texts.append(ans['answer'])
        else:
            answer_texts.append('')
    
    # Generate all pairs
    pairs = list(combinations(range(len(answer_texts)), 2))
    num_pairs = len(pairs)
    
    if num_pairs == 0:
        return 0.0
    
    # Calculate pairwise distances
    total_distance = 0.0
    
    if use_llm and max_workers > 1:
        # Parallel LLM calls
        distances = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {
                executor.submit(
                    call_llm_for_answer_comparison,
                    answer_texts[i],
                    answer_texts[j],
                    llm_client,
                    llm_base_url,
                    llm_model
                ): (i, j)
                for i, j in pairs
            }
            
            for future in as_completed(future_to_pair):
                try:
                    are_equivalent = future.result()
                    distance = 0.0 if are_equivalent else 1.0
                    distances.append(distance)
                except Exception as e:
                    i, j = future_to_pair[future]
                    print(f"    Warning: LLM comparison failed for pair ({i}, {j}): {e}")
                    distances.append(1.0)  # Assume different on error
        
        total_distance = sum(distances)
    else:
        # Sequential processing
        for i, j in pairs:
            if use_llm:
                # Use LLM to compare
                are_equivalent = call_llm_for_answer_comparison(
                    answer_texts[i],
                    answer_texts[j],
                    llm_client=llm_client,
                    llm_base_url=llm_base_url,
                    model=llm_model
                )
                distance = 0.0 if are_equivalent else 1.0
            else:
                # Fallback to exact match
                normalized_i = answer_texts[i].strip().lower()
                normalized_j = answer_texts[j].strip().lower()
                distance = 0.0 if normalized_i == normalized_j else 1.0
            
            total_distance += distance
    
    return total_distance / num_pairs


def metric_2_findings_pairwise(canonical_sets: List[Set[int]]) -> float:
    """
    METRIC 2: Finding-level pairwise stochasticity
    
    Each report has a canonical finding set S[t].
    Distance: normalized cosine distance between binary finding vectors.
    Returns average over all pairs (normalized TV estimator).
    """
    return average_pairwise_distance(canonical_sets)


def metric_4_citation_stochasticity(
    citation_sets: List[Set[int]],
) -> Tuple[float, Dict[str, Any]]:
    """
    METRIC 4: Citation-level stochasticity
    
    Measures the consistency of evidence sourcing by comparing the full
    set of citations used in each run (set-level, not per-finding).
    
    Distance: normalized cosine distance between citation sets.
    
    Args:
        citation_sets: List of sets, where each set contains cluster IDs (canonical citations)
                       for each run/report

    Returns:
        Tuple of (aggregate_citation_stochasticity, details_dict)
    """
    n = len(citation_sets)
    if n < 2:
        return 0.0, {'total_pairs': 0, 'empty_runs': 0}
    
    empty_runs = 0
    for run_citations in citation_sets:
        if not run_citations:
            empty_runs += 1
    
    aggregate_stochasticity = average_pairwise_distance(citation_sets)
    
    details = {
        'total_pairs': n * (n - 1) // 2,
        'empty_runs': empty_runs,
        'citation_counts': [len(s) for s in citation_sets],
    }
    
    return aggregate_stochasticity, details


def build_grouped_citation_sets_from_reports(
    grouped_answers: Dict[str, List[Dict[str, Any]]],
    llm_client: Optional[OpenAI] = None,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
    use_llm: bool = True,
    max_workers: int = 4,
    verbose: bool = True
) -> Tuple[Dict[str, List[Set[int]]], Dict[str, Any]]:
    """
    Extract URLs from final reports using LLM, cluster within each question group,
    and return canonical citation sets per run.
    """
    grouped_citation_sets: Dict[str, List[Set[int]]] = {}
    stats = {
        'missing_trajectories': 0,
        'missing_reports': 0,
        'total_runs': 0,
        'total_urls': 0
    }
    stats_lock = Lock()

    def _bump_stat(key: str, amount: int = 1) -> None:
        with stats_lock:
            stats[key] = stats.get(key, 0) + amount

    def extract_for_answer(answer: Dict[str, Any]) -> List[str]:
        metadata = answer.get('metadata', {}) or {}
        trajectory_file = metadata.get('trajectory_file') or answer.get('trajectory_file')
        source_path = Path(answer.get('source_path', ''))
        if not source_path.exists():
            return []
        trajectory_path = resolve_trajectory_path(source_path, trajectory_file) if trajectory_file else None
        if not trajectory_path or not trajectory_path.exists():
            _bump_stat('missing_trajectories')
            return []
        try:
            with open(trajectory_path, 'r') as f:
                trajectory_data = json.load(f)
        except Exception:
            _bump_stat('missing_trajectories')
            return []
        report_text = extract_final_report_from_trajectory(trajectory_data)
        if not report_text:
            _bump_stat('missing_reports')
            return []
        if use_llm:
            return extract_urls_from_report_llm(
                report_text,
                llm_client=llm_client,
                llm_base_url=llm_base_url,
                model=llm_model
            )
        return _extract_urls_regex(report_text)

    for group_id, answers in grouped_answers.items():
        stats['total_runs'] += len(answers)
        if not answers:
            grouped_citation_sets[group_id] = []
            continue

        urls_per_run: List[List[str]] = [[] for _ in answers]
        if use_llm and max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(extract_for_answer, ans): idx
                    for idx, ans in enumerate(answers)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    urls_per_run[idx] = future.result()
        else:
            for idx, ans in enumerate(answers):
                urls_per_run[idx] = extract_for_answer(ans)

        cluster_map: Dict[str, int] = {}
        next_id = 0
        group_sets: List[Set[int]] = []
        for run_urls in urls_per_run:
            run_set: Set[int] = set()
            for url in run_urls:
                normalized = normalize_url(url)
                if not normalized:
                    continue
                if normalized not in cluster_map:
                    cluster_map[normalized] = next_id
                    next_id += 1
                run_set.add(cluster_map[normalized])
            group_sets.append(run_set)
            stats['total_urls'] += len(run_urls)
        grouped_citation_sets[group_id] = group_sets

        if verbose:
            print(f"  Group '{group_id}': {len(cluster_map)} unique citations across {len(group_sets)} runs")

    return grouped_citation_sets, stats


def calculate_metrics_for_group(
    group_id: str,
    answers: List[Dict[str, Any]],
    canonical_sets: List[Set[int]],
    citation_sets: Optional[List[Set[int]]] = None,
    llm_client: Optional[OpenAI] = None,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
    use_llm: bool = True,
    max_workers: int = 4,
    verbose: bool = True
) -> Dict[str, Any]:
    """Calculate all four metrics for a single question group."""
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Question Group: {group_id}")
        print(f"{'='*70}")
        print(f"  Reports: {len(canonical_sets)}")
        print(f"  Answers: {len(answers)}")
        if canonical_sets:
            for i, s in enumerate(canonical_sets):
                print(f"    Report {i}: {len(s)} canonical findings")
    
    # Calculate metrics
    metric1 = metric_1_answer_pairwise_llm(
        answers, llm_client, llm_base_url, llm_model, use_llm, max_workers
    ) if len(answers) >= 2 else 0.0
    
    metric2 = metric_2_findings_pairwise(canonical_sets) if len(canonical_sets) >= 2 else 0.0
    
    # Calculate citation-level stochasticity (Metric 3)
    metric4 = 0.0
    citation_details = {}
    if citation_sets is not None and len(citation_sets) >= 2:
        metric4, citation_details = metric_4_citation_stochasticity(citation_sets)

    finding_volume_tv = support_size_tv(canonical_sets) if len(canonical_sets) >= 2 else 0.0
    citation_volume_tv = support_size_tv(citation_sets) if citation_sets and len(citation_sets) >= 2 else 0.0
    mean_finding_count = sum(len(s) for s in canonical_sets) / len(canonical_sets) if canonical_sets else 0.0
    mean_citation_count = sum(len(s) for s in citation_sets) / len(citation_sets) if citation_sets else 0.0
    
    if verbose:
        print(f"\n  METRIC 1 (Answer-level):           {metric1:.4f}")
        print(f"  METRIC 2 (Finding-level):          {metric2:.4f}")
        print(f"  METRIC 3 (Citation-level):         {metric4:.4f}")
        print(f"  AUX (Finding volume TV):           {finding_volume_tv:.4f}")
        print(f"  AUX (Citation volume TV):          {citation_volume_tv:.4f}")
        print(f"  AUX (Mean findings):               {mean_finding_count:.2f}")
        print(f"  AUX (Mean citations):              {mean_citation_count:.2f}")
        if citation_details:
            print(f"    Total pairs: {citation_details.get('total_pairs', 0)}")
            print(f"    Empty citation runs: {citation_details.get('empty_runs', 0)}")
    
    return {
        'group_id': group_id,
        'num_reports': len(canonical_sets),
        'num_answers': len(answers),
        'answer_level_pairwise_stochasticity': metric1,
        'finding_level_pairwise_stochasticity': metric2,
        'citation_level_stochasticity': metric4,
        'finding_volume_tv': finding_volume_tv,
        'citation_volume_tv': citation_volume_tv,
        'mean_finding_count': mean_finding_count,
        'mean_citation_count': mean_citation_count,
        'citation_details': citation_details
    }


def main():
    parser = argparse.ArgumentParser(
        description='Calculate stochasticity metrics for deep research systems'
    )
    parser.add_argument(
        'clustered_findings',
        help='Path to clustered findings JSON file'
    )
    parser.add_argument(
        'answers',
        help='Path to answers directory or JSON file'
    )
    parser.add_argument(
        '--theta',
        type=float,
        default=0.5,
        help='(Deprecated) Consensus threshold (no longer used)'
    )
    parser.add_argument(
        '--llm-base-url',
        type=str,
        default=DEFAULT_LLM_BASE_URL,
        help=f'Base URL for local LLM server (default: {DEFAULT_LLM_BASE_URL})'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_LLM_MODEL,
        help=f'LLM model name (default: {DEFAULT_LLM_MODEL})'
    )
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM comparison, use exact string matching'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file for results (optional)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of parallel workers for LLM calls (default: 4, set to 1 for sequential)'
    )
    parser.add_argument(
        '--accuracy',
        action='store_true',
        help='Calculate accuracy by comparing with ground truth (requires --reference)'
    )
    parser.add_argument(
        '--reference',
        type=str,
        help='Path to reference QA pairs JSON file for accuracy calculation'
    )
    parser.add_argument(
        '--accuracy-model',
        type=str,
        default=DEFAULT_LLM_MODEL,
        help=f'LLM model for accuracy evaluation (default: {DEFAULT_LLM_MODEL})'
    )
    
    # Legacy arguments for backward compatibility (deprecated)
    parser.add_argument(
        '--llm-provider',
        type=str,
        default='local',
        help='DEPRECATED: Use --llm-base-url instead. Kept for backward compatibility.'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='DEPRECATED: Not needed for local LLM server'
    )
    
    args = parser.parse_args()
    
    # Validate accuracy arguments
    if args.accuracy and not args.reference:
        parser.error("--accuracy requires --reference to be specified")
    
    verbose = not args.quiet
    use_llm = not args.no_llm
    
    # Initialize LLM client
    llm_client = None
    if use_llm:
        llm_client = create_openai_client(args.llm_base_url)
        if verbose:
            print(f"LLM client initialized: {args.llm_base_url}")
            print(f"LLM model: {args.model}")
    
    # Load data
    if verbose:
        print("Loading data...")
    
    clustered_data = load_clustered_findings(args.clustered_findings)
    grouped_answers = load_answers_grouped(args.answers)
    
    if verbose:
        print(f"  Found {len(grouped_answers)} question groups")
        for group_id, answers in grouped_answers.items():
            print(f"    Group '{group_id}': {len(answers)} answers")
    
    # Extract canonical sets and group them to match answer groups
    all_canonical_sets = extract_canonical_finding_sets(clustered_data)
    
    # Get source filenames from metadata if available
    source_filenames = clustered_data.get('metadata', {}).get('source_filenames', [])
    
    if source_filenames and verbose:
        print(f"  Using explicit filename matching with {len(source_filenames)} source files")
    elif verbose:
        print("  Warning: No source_filenames in metadata, using position-based matching")
    
    grouped_canonical_sets = group_canonical_sets_by_answers(
        all_canonical_sets, grouped_answers, source_filenames
    )
    
    if verbose and len(grouped_canonical_sets) != len(grouped_answers):
        print(f"Warning: Mismatch between answer groups ({len(grouped_answers)}) and canonical set groups ({len(grouped_canonical_sets)})")
    
    # Build citation sets from final reports using LLM extraction
    grouped_citation_sets, citation_stats = build_grouped_citation_sets_from_reports(
        grouped_answers,
        llm_client=llm_client,
        llm_base_url=args.llm_base_url,
        llm_model=args.model,
        use_llm=use_llm,
        max_workers=args.max_workers,
        verbose=verbose
    )
    if verbose:
        print(
            f"  Citation extraction: {citation_stats.get('total_urls', 0)} URLs "
            f"across {citation_stats.get('total_runs', 0)} runs "
            f"(missing trajectories: {citation_stats.get('missing_trajectories', 0)}, "
            f"missing reports: {citation_stats.get('missing_reports', 0)})"
        )
    
    if verbose:
        print(f"\nMax workers for LLM calls: {args.max_workers}")
    
    # Calculate metrics for each group
    group_results = []
    
    # Process groups
    group_ids = sorted(grouped_answers.keys())
    
    for group_id in group_ids:
        answers = grouped_answers[group_id]
        canonical_sets = grouped_canonical_sets.get(group_id, [])
        
        if len(canonical_sets) != len(answers) and verbose:
            print(f"Warning: Group '{group_id}' has {len(answers)} answers but {len(canonical_sets)} canonical sets")
        
        citation_sets = grouped_citation_sets.get(group_id, [])
        if len(citation_sets) != len(answers) and verbose:
            print(f"Warning: Group '{group_id}' has {len(answers)} answers but {len(citation_sets)} citation sets")

        result = calculate_metrics_for_group(
            group_id=group_id,
            answers=answers,
            canonical_sets=canonical_sets,
            citation_sets=citation_sets,
            llm_client=llm_client,
            llm_base_url=args.llm_base_url,
            llm_model=args.model,
            use_llm=use_llm,
            max_workers=args.max_workers,
            verbose=verbose
        )
        
        group_results.append(result)
    
    # Calculate aggregate metrics (average across all groups)
    if len(group_results) > 1:
        avg_metric1 = sum(r['answer_level_pairwise_stochasticity'] for r in group_results) / len(group_results)
        avg_metric2 = sum(r['finding_level_pairwise_stochasticity'] for r in group_results) / len(group_results)
        avg_metric3 = sum(r['citation_level_stochasticity'] for r in group_results) / len(group_results)
        avg_finding_volume = sum(r['finding_volume_tv'] for r in group_results) / len(group_results)
        avg_citation_volume = sum(r['citation_volume_tv'] for r in group_results) / len(group_results)
        avg_mean_findings = sum(r['mean_finding_count'] for r in group_results) / len(group_results)
        avg_mean_citations = sum(r['mean_citation_count'] for r in group_results) / len(group_results)
        
        if verbose:
            print(f"\n{'='*70}")
            print("AGGREGATE METRICS (averaged across all question groups)")
            print(f"{'='*70}")
            print(f"  METRIC 1 (Answer-level):           {avg_metric1:.4f}")
            print(f"  METRIC 2 (Finding-level):          {avg_metric2:.4f}")
            print(f"  METRIC 3 (Citation-level):         {avg_metric3:.4f}")
            print(f"  AUX (Finding volume TV):           {avg_finding_volume:.4f}")
            print(f"  AUX (Citation volume TV):          {avg_citation_volume:.4f}")
            print(f"  AUX (Mean findings):               {avg_mean_findings:.2f}")
            print(f"  AUX (Mean citations):              {avg_mean_citations:.2f}")
            print(f"{'='*70}")
        
        aggregate_results = {
            'answer_level_pairwise_stochasticity': avg_metric1,
            'finding_level_pairwise_stochasticity': avg_metric2,
            'citation_level_stochasticity': avg_metric3,
            'finding_volume_tv': avg_finding_volume,
            'citation_volume_tv': avg_citation_volume,
            'mean_finding_count': avg_mean_findings,
            'mean_citation_count': avg_mean_citations
        }
    else:
        aggregate_results = None
    
    # Calculate accuracy if requested
    accuracy_results = None
    if args.accuracy and args.reference:
        reference_qa = load_reference_qa_pairs(args.reference)
        accuracy_model = args.accuracy_model if args.accuracy_model else args.model
        if verbose:
            print(f"\nLoaded {len(reference_qa)} reference QA pairs from {args.reference}")
            print(f"Using model for accuracy: {accuracy_model}")
        
        accuracy_results = calculate_accuracy(
            grouped_answers=grouped_answers,
            reference_qa=reference_qa,
            llm_client=llm_client,
            llm_base_url=args.llm_base_url,
            llm_model=accuracy_model,
            use_llm=use_llm,
            max_workers=args.max_workers,
            verbose=verbose
        )
        
        if verbose:
            print(f"\n{'='*70}")
            print("ACCURACY RESULTS")
            print(f"{'='*70}")
            print(f"  Overall Accuracy: {accuracy_results['overall_accuracy']:.2%}")
            print(f"  Correct: {accuracy_results['total_correct']}/{accuracy_results['total_questions']}")
            print(f"{'='*70}")
    
    # Prepare output
    output = {
        'llm_base_url': args.llm_base_url if use_llm else 'none',
        'llm_model': args.model if use_llm else 'none',
        'num_question_groups': len(group_results),
        'per_group_metrics': group_results,
        'aggregate_metrics': aggregate_results,
        'accuracy': accuracy_results
    }
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        if verbose:
            print(f"\nResults saved to {args.output}")
    
    return output


if __name__ == '__main__':
    main()

