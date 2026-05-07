#!/usr/bin/env python3
"""
Script to extract factual claims/answers/atomic facts from deep research trajectory reports using a local OpenAI-compatible LLM API.

This script:
1. Reads trajectory JSON files containing research reports
2. Extracts the research question and final report
3. Calls local LLM server to extract claims, QA answers, or atomic facts
4. Saves the extracted data as JSON files

Modes:
- claims: Extract factual claims with citations (default)
- qa_answers: Extract only answers for QA tasks
- atomic_facts: Decompose claims into atomic facts
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
from openai import OpenAI
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Default configuration for local LLM server
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:30000/v1")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")


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


# Task descriptions for different extraction modes

# Mode 1: Claims extraction (original)
CLAIMS_TASK_DESCRIPTION = """## Task Description
Extract all factual claims from the provided report. Each claim should be a factual statement that
can be verified. Claims may or may not have supporting citations.

## Input
A Research Question and a complete report containing factual claims, some of which may have
citation markers and corresponding URLs (either inline or in a reference section).

## Output Requirements
• Extract each distinct factual claim throughout the entire report
• For each claim, output a JSON object with:
  – The exact claim text as a string
  – The original text from the report containing this claim (context)
  – The corresponding citation URL as source (if a citation marker directly follows the claim)
• If a claim has a citation marker directly following it, return the supporting URL as source
• If a claim does not have a citation marker directly following it, return an empty string for source
• Ensure all string values are properly escaped for valid JSON format (e.g. Replace internal quotation
  marks (") with escaped quotation marks (\\")) in the claim and context
• Return a JSON array containing all claim objects

## Format Specification
[
  {
    "claim": "The exact statement representing a factual claim",
    "context": "The original sentence or passage from the report containing this claim",
    "source": "https://example.com/source1"
  },
  {
    "claim": "Another factual statement without direct citation",
    "context": "The original sentence or passage from the report containing this claim",
    "source": ""
  }
]

## Example of Pronoun Resolution
INCORRECT: "He worked as a probation officer for eight years."
CORRECT: "Ken Walibora worked as a probation officer for eight years."

INCORRECT: "She won the Nobel Prize in 2020."
CORRECT: "Marie Curie won the Nobel Prize in 2020."

INCORRECT: "The company launched its product in January."
CORRECT: "Apple launched its product in January." OR "Apple launched the iPhone in January."

## Guidelines for Claim Identification
1. A claim should be a complete, standalone factual statement that is unambiguous when read in isolation
2. Replace all pronouns (he, she, it, they, etc.) with the actual entity names to make claims self-contained
3. Each claim must be understandable without requiring context from other claims or the report
4. Maintain factual accuracy while ensuring clarity - if "he" refers to "Ken Walibora", replace it with "Ken Walibora"
5. Extract all factual claims regardless of whether they have citation support
6. Only consider to map citation markers (numbers, author names, etc.) to their corresponding URLs in
   the references section when it directly follow the claim statement
7. Exclude opinions, speculations, or methodological descriptions
8. Extract the context passage containing each claim for verification purposes
9. If multiple claims are associated with the same citation, extract them as separate entries

## Citation URL Mapping
• If URLs appear directly after claims, use those URLs directly
• Citation markers (e.g. follows a number or [number]) must directly follow the claim to be considered
  as supporting that claim
• If claims use citation markers that reference a bibliography or reference section, locate the correspond-
  ing URLs in that section
• If a claim has no directly following citation marker, use an empty string for source
"""

# Mode 2: QA answers extraction
QA_ANSWERS_TASK_DESCRIPTION = """## Task Description
Extract the answer to the research question from the provided report.

## Input
A Research Question and a complete report containing the answer.

## Output Requirements
• Extract the direct answer to the research question
• Provide supporting evidence/context from the report
• Return a JSON object with the answer and supporting context

## Format Specification
{
  "question": "The research question",
  "answer": "The direct answer extracted from the report",
  "supporting_context": "Key passages from the report that support this answer"
}

## Guidelines
1. Focus on directly answering the research question
2. Be concise but comprehensive
3. Include relevant evidence and context
4. Maintain factual accuracy 
"""

# Mode 3: Atomic facts decomposition
ATOMIC_FACTS_TASK_DESCRIPTION = """You are given a factual statement (a claim) from a technical report.
Break the claim down into independent, minimal atomic facts that can each be
verified in isolation. Keep each atomic fact short, declarative, and free of
conjunctions when possible. Avoid duplicating the same content in multiple ways
unless it clarifies distinct atomic facts (e.g., subject + membership vs subject + role).

CRITICAL: Each atomic fact must be self-contained and unambiguous:
- Replace ALL pronouns (he, she, it, they, etc.) with actual entity names
- Each fact must be understandable in complete isolation without any context

Format:
- Return a JSON array of strings. Each string is one atomic fact.
- Do not include any extra commentary or Markdown. Only return the JSON array.

Examples:
Input: "Leonard Bernstein was an American composer, conductor, and musical director."
Output: [
  "Leonard Bernstein was an American.",
  "Leonard Bernstein was a composer.",
  "Leonard Bernstein was a conductor.",
  "Leonard Bernstein was a musical director."
]

Input: "Zhang Ziyi currently stars in the romantic comedy series, Love and Destiny, which premiered in 2019."
Output: [
  "Zhang Ziyi currently stars in Love and Destiny.",
  "Love and Destiny is a romantic comedy series.",
  "Love and Destiny premiered in 2019."
]

Input: "During his professional career, McCoy played for the Broncos, the San Diego Chargers, the Minnesota Vikings, and the Jacksonville Jaguars."
Output: [
  "McCoy played for the Broncos.",
  "McCoy played for the Broncos during his professional career.",
  "McCoy played for the San Diego Chargers.",
  "McCoy played for the San Diego Chargers during his professional career.",
  "McCoy played for the Minnesota Vikings.",
  "McCoy played for the Minnesota Vikings during his professional career.",
  "McCoy played for the Jacksonville Jaguars.",
  "McCoy played for the Jacksonville Jaguars during his professional career."
]

Input: "The EU approved the AI Act in 2024 and introduced new compliance requirements."
Output: [
  "The EU approved the AI Act in 2024.",
  "The AI Act introduced new compliance requirements."
]

Input: "The Amazon River is the largest by discharge and flows into the Atlantic Ocean."
Output: [
  "The Amazon River is the largest river by discharge.",
  "The Amazon River flows into the Atlantic Ocean."
]

Now decompose the following claim into atomic facts and return only a JSON array of strings:"""


def extract_research_question(trajectory_data: Dict[str, Any]) -> Optional[str]:
    """
    Extract the research question from the trajectory data.
    
    Supports multiple formats:
    - Direct "research_question" field (DeepResearch format)
    - First human message in trajectory_steps
    
    Args:
        trajectory_data: The parsed JSON trajectory data
        
    Returns:
        The research question as a string, or None if not found
    """
    # First, check for direct research_question field (DeepResearch format)
    if "research_question" in trajectory_data:
        return trajectory_data["research_question"]
    
    # Fall back to looking for the first human message in trajectory steps
    trajectory_steps = trajectory_data.get("trajectory_steps", [])
    
    for step in trajectory_steps:
        input_state = step.get("input_state", {})
        messages = input_state.get("messages", [])
        
        for message in messages:
            if message.get("type") == "human":
                return message.get("content", "")
    
    return None


def extract_references_from_report(report_text: str) -> str:
    """
    Extract the references/sources section from the report.
    
    Args:
        report_text: The complete report text
        
    Returns:
        The references section as a string
    """
    # Look for common reference section headers
    patterns = [
        r"##\s*Sources\s*\n(.*?)(?=\n##|\Z)",
        r"##\s*References\s*\n(.*?)(?=\n##|\Z)",
        r"#\s*Sources\s*\n(.*?)(?=\n#|\Z)",
        r"#\s*References\s*\n(.*?)(?=\n#|\Z)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, report_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If no explicit section found, try to extract inline citations
    # Look for numbered citations like [1], [2], etc. and their URLs
    citations = re.findall(r'\[(\d+)\][^\n]*?https?://[^\s\)]+', report_text)
    if citations:
        return "Citations found in text body"
    
    return "No explicit references section found"


def create_extraction_prompt(question: str, report: str, references: str, mode: str = "claims") -> str:
    """
    Create the full prompt based on the extraction mode.
    
    Args:
        question: The research question
        report: The complete report text
        references: The references section
        mode: Extraction mode ("claims", "qa_answers", or "atomic_facts")
        
    Returns:
        The complete prompt string
    """
    if mode == "qa_answers":
        prompt = f"""{QA_ANSWERS_TASK_DESCRIPTION}

Research Question: {question}

Report: {report}
"""
    elif mode == "atomic_facts":
        # For atomic facts, we first extract claims then decompose them
        prompt = f"""{CLAIMS_TASK_DESCRIPTION}

Please extract all claims from the following report and provide them in the specified JSON format:

Research Question: {question}

Response Content: {report}

References: {references}
"""
    else:  # mode == "claims"
        prompt = f"""{CLAIMS_TASK_DESCRIPTION}

Please extract all claims from the following report and provide them in the specified JSON format:

Research Question: {question}

Response Content: {report}

References: {references}
"""
    return prompt


def decompose_claim_to_atomic_facts(
    client: OpenAI,
    claim: str,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.1,
    max_retries: int = 3
) -> List[str]:
    """
    Decompose a single claim into atomic facts using a local OpenAI-compatible LLM API.
    
    Args:
        client: OpenAI client instance (pointing to local server)
        claim: The claim to decompose
        model: Model name to use
        temperature: Temperature for the API call
        max_retries: Maximum number of retry attempts
        
    Returns:
        List of atomic facts
    """
    prompt = f"{ATOMIC_FACTS_TASK_DESCRIPTION}\n\n{claim}"
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=4096,  # Sufficient for atomic facts decomposition
                seed=42  # Required by SGLang when temperature=0
            )
            
            result_text = response.choices[0].message.content
            # Parse the JSON array
            atomic_facts = json.loads(result_text)
            
            if not isinstance(atomic_facts, list):
                atomic_facts = [result_text]
            
            return atomic_facts
            
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            try:
                json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
                if json_match:
                    atomic_facts = json.loads(json_match.group())
                    return atomic_facts
            except:
                pass
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
            else:
                print(f"Error parsing atomic facts response: {result_text[:200]}...")
                return [claim]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
            else:
                print(f"Error decomposing claim to atomic facts after {max_retries} retries: {e}")
                return [claim]  # Return original claim if decomposition fails
    
    return [claim]


def extract_data_with_llm(
    client: OpenAI,
    question: str,
    report: str,
    references: str,
    mode: str = "claims",
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.1,
    max_retries: int = 3
) -> Any:
    """
    Extract data using a local OpenAI-compatible LLM API based on the mode.
    
    Args:
        client: OpenAI client instance (pointing to local server)
        question: The research question
        report: The complete report text
        references: The references section
        mode: Extraction mode ("claims", "qa_answers", or "atomic_facts")
        model: Model name to use
        temperature: Temperature for the API call
        max_retries: Maximum number of retry attempts
        
    Returns:
        Extracted data (format depends on mode)
    """
    prompt = create_extraction_prompt(question, report, references, mode)
    
    if mode == "qa_answers":
        system_content = """You are an expert at answering questions based on research reports. You provide clear, accurate answers with supporting evidence. 

CRITICAL: Return ONLY valid JSON. Ensure all strings are properly escaped:
- Escape quotes as \"
- Escape newlines as \\n
- Escape backslashes as \\\\
Do not include any text before or after the JSON."""
    else:
        system_content = """You are an expert at extracting factual claims from reports. You carefully identify claims, their context, and associated citations.

CRITICAL REQUIREMENTS:
1. Make each claim self-contained and unambiguous - replace ALL pronouns (he, she, it, they, etc.) with the actual entity names
2. Each claim must be understandable in complete isolation without any context
3. Return ONLY valid JSON. Ensure all strings are properly escaped:
   - Escape quotes as \"
   - Escape newlines as \\n
   - Escape backslashes as \\\\
Do not include any text before or after the JSON."""
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_content
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=8192,  # Increased to prevent truncation
                seed=42  # Required by SGLang when temperature=0
            )
            
            result_text = response.choices[0].message.content
            
            # Try to extract JSON from markdown code blocks if present
            json_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(1)
            
            # Try to parse JSON with error handling
            try:
                result_data = json.loads(result_text)
            except json.JSONDecodeError as e:
                print(f"  ⚠️  JSON parsing error: {e}")
                print(f"  Attempting to repair JSON...")
                
                # Save problematic response for debugging
                debug_file = Path(f"debug_response_{mode}.txt")
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(result_text)
                print(f"  Saved raw response to: {debug_file}")
                
                # Try to fix common JSON issues
                result_text_fixed = result_text
                
                # Remove trailing commas before closing brackets
                result_text_fixed = re.sub(r',(\s*[}\]])', r'\1', result_text_fixed)
                
                # Try to fix unterminated strings at the end
                # If we have an incomplete JSON array, try to close it properly
                if '"context":' in result_text_fixed or '"claim":' in result_text_fixed:
                    # Find the last complete claim object
                    last_complete = result_text_fixed.rfind('},')
                    if last_complete > 0:
                        # Truncate to last complete object and close the array
                        result_text_fixed = result_text_fixed[:last_complete + 1] + '\n]'
                        print(f"  ℹ️  Truncated to last complete claim object")
                
                # Try parsing again
                try:
                    result_data = json.loads(result_text_fixed)
                    print(f"  ✓ JSON repaired successfully (regex)")
                except json.JSONDecodeError:
                    # Use LLM to repair the malformed JSON
                    print(f"  ℹ️  Attempting LLM-based JSON repair...")
                    try:
                        repair_response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a JSON repair tool. Your ONLY job is to take malformed JSON and output valid JSON. Do NOT add any explanation, markdown fences, or extra text. Output ONLY the corrected JSON."
                                },
                                {
                                    "role": "user",
                                    "content": f"Fix this malformed JSON and return ONLY valid JSON:\n\n{result_text[:6000]}"
                                }
                            ],
                            temperature=0.0,
                            max_tokens=8192,
                            seed=42
                        )
                        repaired_text = repair_response.choices[0].message.content.strip()
                        # Strip markdown fences if present
                        repair_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', repaired_text)
                        if repair_match:
                            repaired_text = repair_match.group(1)
                        result_data = json.loads(repaired_text)
                        print(f"  ✓ JSON repaired successfully (LLM)")
                    except (json.JSONDecodeError, Exception) as llm_err:
                        print(f"  ✗ LLM repair also failed: {llm_err}")
                        if attempt < max_retries - 1:
                            print(f"  Retrying... (attempt {attempt + 2}/{max_retries})")
                            time.sleep(1 * (attempt + 1))
                            continue
                        print(f"  ✗ Could not repair JSON automatically")
                        raise
            
            # Handle different modes
            if mode == "qa_answers":
                return result_data
            
            else:  # mode == "claims"
                # Handle different response formats
                if isinstance(result_data, list):
                    claims = result_data
                elif "claims" in result_data:
                    claims = result_data["claims"]
                elif "data" in result_data:
                    claims = result_data["data"]
                else:
                    for value in result_data.values():
                        if isinstance(value, list):
                            claims = value
                            break
                    else:
                        claims = [result_data]
                
                return claims
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Error during LLM call (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(1 * (attempt + 1))
            else:
                print(f"Error during LLM call after {max_retries} retries: {e}")
                raise
    
    return []


def process_claims_file_for_atomic_facts(
    client: OpenAI,
    claims_file: Path,
    output_dir: Path,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.1,
    print_lock: Optional[Lock] = None
) -> None:
    """
    Process a claims file and decompose claims into atomic facts.
    
    Args:
        client: OpenAI client instance (pointing to local server)
        claims_file: Path to the claims JSON file
        output_dir: Directory to save the output
        model: Model name to use
        temperature: Temperature for the API call
        print_lock: Optional lock for thread-safe printing
    """
    def safe_print(msg):
        """Thread-safe printing."""
        if print_lock:
            with print_lock:
                print(msg)
        else:
            print(msg)
    
    safe_print(f"\nProcessing claims file: {claims_file.name}")
    
    # Read the claims file
    with open(claims_file, 'r', encoding='utf-8') as f:
        claims_data = json.load(f)
    
    session_id = claims_data.get("session_id", claims_file.stem)
    research_question = claims_data.get("research_question", "")
    claims = claims_data.get("claims", [])
    
    if not claims:
        safe_print(f"  ⚠️  Warning: No claims found in {claims_file.name}")
        return
    
    safe_print(f"  Research Question: {research_question[:100]}...")
    safe_print(f"  Number of claims to decompose: {len(claims)}")
    safe_print(f"  Decomposing claims to atomic facts using {model}...")
    
    # Decompose each claim into atomic facts
    all_atomic_facts = []
    for i, claim_obj in enumerate(claims, 1):
        claim_text = claim_obj.get("claim", "")
        if claim_text:
            if not print_lock:
                print(f"    Processing claim {i}/{len(claims)}...", end="\r")
            atomic_facts = decompose_claim_to_atomic_facts(client, claim_text, model, temperature)
            all_atomic_facts.append({
                "original_claim": claim_text,
                "context": claim_obj.get("context", ""),
                "source": claim_obj.get("source", ""),
                "atomic_facts": atomic_facts
            })
    
    safe_print(f"  ✓ Decomposed {len(claims)} claims into atomic facts")
    
    # Save atomic facts to output file
    output_file = output_dir / f"atomic_facts_{session_id}.json"
    output_data = {
        "session_id": session_id,
        "research_question": research_question,
        "num_original_claims": len(all_atomic_facts),
        "atomic_facts_data": all_atomic_facts,
        "metadata": {
            "claims_file": str(claims_file),
            "num_claims_processed": len(claims),
            "model": model,
            "temperature": temperature,
            "mode": "atomic_facts"
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    safe_print(f"  ✓ Saved to: {output_file}")


def process_trajectory_file(
    client: OpenAI,
    trajectory_file: Path,
    output_dir: Path,
    mode: str = "claims",
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.1,
    print_lock: Optional[Lock] = None
) -> None:
    """
    Process a single trajectory file and extract data based on mode.
    
    Args:
        client: OpenAI client instance (pointing to local server)
        trajectory_file: Path to the trajectory JSON file
        output_dir: Directory to save the output
        mode: Extraction mode ("claims", "qa_answers", or "atomic_facts")
        model: Model name to use
        temperature: Temperature for the API call
        print_lock: Optional lock for thread-safe printing
    """
    def safe_print(msg):
        """Thread-safe printing."""
        if print_lock:
            with print_lock:
                print(msg)
        else:
            print(msg)
    
    safe_print(f"\nProcessing: {trajectory_file.name}")
    
    # Read the trajectory file
    with open(trajectory_file, 'r', encoding='utf-8') as f:
        trajectory_data = json.load(f)
    
    # Extract components
    session_id = trajectory_data.get("session_id", trajectory_file.stem)
    question = extract_research_question(trajectory_data)
    report = trajectory_data.get("final_report", "")
    
    if not question:
        safe_print(f"  ⚠️  Warning: Could not extract research question from {trajectory_file.name}")
        question = "Research question not found in trajectory"
    
    if not report:
        safe_print(f"  ⚠️  Warning: No final report found in {trajectory_file.name}")
        return
    
    references = extract_references_from_report(report)
    
    safe_print(f"  Research Question: {question[:100]}...")
    safe_print(f"  Report Length: {len(report)} characters")
    safe_print(f"  Extracting data using {model} in mode: {mode}...")
    
    # Extract data using local LLM API (only claims or qa_answers here)
    extracted_data = extract_data_with_llm(
        client=client,
        question=question,
        report=report,
        references=references,
        mode=mode,
        model=model,
        temperature=temperature
    )
    
    # Prepare output based on mode
    if mode == "qa_answers":
        safe_print(f"  ✓ Extracted QA answer")
        output_file = output_dir / f"qa_answer_{session_id}.json"
        output_data = {
            "session_id": session_id,
            "research_question": question,
            "answer_data": extracted_data,
            "metadata": {
                "trajectory_file": str(trajectory_file),
                "report_length": len(report),
                "model": model,
                "temperature": temperature,
                "mode": mode
            }
        }
    else:  # mode == "claims"
        safe_print(f"  ✓ Extracted {len(extracted_data)} claims")
        output_file = output_dir / f"claims_{session_id}.json"
        output_data = {
            "session_id": session_id,
            "research_question": question,
            "num_claims": len(extracted_data),
            "claims": extracted_data,
            "metadata": {
                "trajectory_file": str(trajectory_file),
                "report_length": len(report),
                "model": model,
                "temperature": temperature,
                "mode": mode
            }
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    safe_print(f"  ✓ Saved to: {output_file}")


def main():
    """Main function to orchestrate extraction based on mode."""
    parser = argparse.ArgumentParser(
        description="Extract claims/answers/atomic facts from deep research trajectory reports using a local OpenAI-compatible LLM API"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./reports",
        help="Directory containing trajectory JSON files (or claims files for atomic_facts mode)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./claims",
        help="Directory to save extracted data"
    )
    parser.add_argument(
        "--trajectory-file",
        type=str,
        help="Process a specific trajectory file instead of all files in input-dir"
    )
    parser.add_argument(
        "--claims-file",
        type=str,
        help="Process a specific claims file (for atomic_facts mode only)"
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=DEFAULT_LLM_BASE_URL,
        help=f"Base URL for local LLM server (default: {DEFAULT_LLM_BASE_URL})"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["claims", "qa_answers", "atomic_facts"],
        default="claims",
        help="Extraction mode: 'claims' (default), 'qa_answers', or 'atomic_facts'"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help=f"LLM model name (default: {DEFAULT_LLM_MODEL})"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for API calls (default: 0.1)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers for processing files (default: 4, set to 1 for sequential)"
    )
    
    # Legacy argument for backward compatibility
    parser.add_argument(
        "--api-key",
        type=str,
        help="DEPRECATED: Not needed for local LLM server"
    )
    
    args = parser.parse_args()
    
    # Setup local OpenAI-compatible client
    client = create_openai_client(args.llm_base_url)
    
    # Setup directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("DATA EXTRACTION FROM DEEP RESEARCH REPORTS")
    print("=" * 80)
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Mode: {args.mode}")
    print(f"LLM Base URL: {args.llm_base_url}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max Workers: {args.max_workers}")
    
    # Create print lock for thread-safe printing
    print_lock = Lock() if args.max_workers > 1 else None
    
    # Handle atomic_facts mode differently - it uses claims files as input
    if args.mode == "atomic_facts":
        if args.claims_file:
            # Process single claims file
            claims_file = Path(args.claims_file)
            if not claims_file.exists():
                print(f"Error: File not found: {claims_file}")
                return 1
            
            process_claims_file_for_atomic_facts(
                client=client,
                claims_file=claims_file,
                output_dir=output_dir,
                model=args.model,
                temperature=args.temperature,
                print_lock=print_lock
            )
        else:
            # Process all claims files in input directory
            claims_files = list(input_dir.glob("claims_*.json"))
            
            if not claims_files:
                print(f"\nNo claims files found in {input_dir}")
                print("Hint: For atomic_facts mode, use --input-dir to point to the directory containing claims_*.json files")
                print("Or use --claims-file to specify a specific claims file")
                return 1
            
            print(f"\nFound {len(claims_files)} claims file(s)")
            
            # Process files in parallel or sequentially
            if args.max_workers > 1:
                print(f"Processing files in parallel with {args.max_workers} workers...")
                with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                    # Submit all tasks
                    future_to_file = {
                        executor.submit(
                            process_claims_file_for_atomic_facts,
                            client,
                            claims_file,
                            output_dir,
                            args.model,
                            args.temperature,
                            print_lock
                        ): claims_file
                        for claims_file in claims_files
                    }
                    
                    # Process completed tasks
                    for future in as_completed(future_to_file):
                        claims_file = future_to_file[future]
                        try:
                            future.result()
                        except Exception as e:
                            if print_lock:
                                with print_lock:
                                    print(f"  ✗ Error processing {claims_file.name}: {e}")
                            else:
                                print(f"  ✗ Error processing {claims_file.name}: {e}")
            else:
                print("Processing files sequentially...")
                for claims_file in claims_files:
                    try:
                        process_claims_file_for_atomic_facts(
                            client=client,
                            claims_file=claims_file,
                            output_dir=output_dir,
                            model=args.model,
                            temperature=args.temperature,
                            print_lock=print_lock
                        )
                    except Exception as e:
                        print(f"  ✗ Error processing {claims_file.name}: {e}")
                        continue
    else:
        # Process trajectory files for claims and qa_answers modes
        if args.trajectory_file:
            # Process single file
            trajectory_file = Path(args.trajectory_file)
            if not trajectory_file.exists():
                print(f"Error: File not found: {trajectory_file}")
                return 1
            
            process_trajectory_file(
                client=client,
                trajectory_file=trajectory_file,
                output_dir=output_dir,
                mode=args.mode,
                model=args.model,
                temperature=args.temperature,
                print_lock=print_lock
            )
        else:
            # Process all trajectory files in input directory
            trajectory_files = list(input_dir.glob("deep_research_trajectory_*.json"))
            
            if not trajectory_files:
                print(f"\nNo trajectory files found in {input_dir}")
                return 1
            
            print(f"\nFound {len(trajectory_files)} trajectory file(s)")
            
            # Process files in parallel or sequentially
            if args.max_workers > 1:
                print(f"Processing files in parallel with {args.max_workers} workers...")
                with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                    # Submit all tasks
                    future_to_file = {
                        executor.submit(
                            process_trajectory_file,
                            client,
                            trajectory_file,
                            output_dir,
                            args.mode,
                            args.model,
                            args.temperature,
                            print_lock
                        ): trajectory_file
                        for trajectory_file in trajectory_files
                    }
                    
                    # Process completed tasks
                    for future in as_completed(future_to_file):
                        trajectory_file = future_to_file[future]
                        try:
                            future.result()
                        except Exception as e:
                            if print_lock:
                                with print_lock:
                                    print(f"  ✗ Error processing {trajectory_file.name}: {e}")
                            else:
                                print(f"  ✗ Error processing {trajectory_file.name}: {e}")
            else:
                print("Processing files sequentially...")
                for trajectory_file in trajectory_files:
                    try:
                        process_trajectory_file(
                            client=client,
                            trajectory_file=trajectory_file,
                            output_dir=output_dir,
                            mode=args.mode,
                            model=args.model,
                            temperature=args.temperature,
                            print_lock=print_lock
                        )
                    except Exception as e:
                        print(f"  ✗ Error processing {trajectory_file.name}: {e}")
                        continue
    
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())

