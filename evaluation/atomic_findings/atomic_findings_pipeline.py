#!/usr/bin/env python3
"""
Atomic Findings Embedding Pipeline

Extracts embeddings for atomic findings from Deep Research Agent atomic_facts files
and performs semantic clustering using a local OpenAI-compatible embedding API.

Usage:
    python atomic_findings_pipeline.py <path_to_atomic_facts_dir> [--threshold 0.82] [--output results.json]
    python atomic_findings_pipeline.py file1.json file2.json [--threshold 0.82] [--output results.json]
"""

import json
import argparse
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Set, Tuple, Any, Optional
import numpy as np
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Default configuration for local LLM server
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:30000/v1")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
DEFAULT_LLM_API_KEY = os.getenv("TOGETHER_API_KEY", "EMPTY")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct")


class DisjointSet:
    """Union-Find data structure with path compression and union by rank."""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """Unite sets using union by rank."""
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return False
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        return True
    
    def get_clusters(self) -> Dict[int, List[int]]:
        """Get all clusters."""
        clusters = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)
        return clusters


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two embedding vectors."""
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    
    dot_product = np.dot(vec1_np, vec2_np)
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def load_atomic_facts_file(filepath: str) -> Dict[str, Any]:
    """Load atomic facts from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_group_from_filename(filename: str) -> Optional[str]:
    """
    Extract group identifier from filename.
    
    Supports multiple formats:
    1. DeepResearch format: 'atomic_facts_deepresearch_iter1_800888c0.json' -> '800888c0'
    2. Alternative format: 'atomic_facts_0_1.json' -> '0'
    """
    import re
    
    # Format 1: DeepResearch format with iter and hex hash
    # Pattern: *_iter{N}_{hexhash}.json
    match = re.search(r'_iter\d+_([a-f0-9]+)\.json', filename)
    if match:
        return match.group(1)  # Hex hash is group/question ID
    
    # Format 2: Legacy format with _X_Y where X is group, Y is run
    match = re.search(r'atomic_facts_(\d+)_\d+\.json', filename)
    if match:
        return match.group(1)
    
    return None


def extract_all_atomic_findings(atomic_facts_files: List[Dict[str, Any]], source_filenames: Optional[List[str]] = None) -> Tuple[List[str], List[Tuple[int, int]], List[Optional[str]], List[List[str]]]:
    """
    Extract all atomic findings from multiple atomic_facts files into a flat list.
    
    Returns:
        - List of all atomic finding strings
        - List of (file_idx, claim_idx) mappings for each finding
        - List of group identifiers for each finding
        - List of citation lists for each finding (source URLs from the original claim)
    """
    all_findings: List[str] = []
    finding_origins: List[Tuple[int, int]] = []
    finding_groups: List[Optional[str]] = []
    finding_citations: List[List[str]] = []  # Citations for each finding
    
    for file_idx, atomic_facts_data in enumerate(atomic_facts_files):
        # Extract group from filename if available
        group_id = None
        if source_filenames and file_idx < len(source_filenames):
            group_id = extract_group_from_filename(source_filenames[file_idx])
        
        claims_data = atomic_facts_data.get('atomic_facts_data', [])
        
        for claim_idx, claim_data in enumerate(claims_data):
            atomic_facts = claim_data.get('atomic_facts', [])
            # Get citation/source from the original claim
            source = claim_data.get('source', '')
            citations = [source] if source else []
            
            for fact in atomic_facts:
                all_findings.append(fact)
                finding_origins.append((file_idx, claim_idx))
                finding_groups.append(group_id)
                finding_citations.append(citations)
    
    return all_findings, finding_origins, finding_groups, finding_citations


def get_embeddings(
    texts: List[str], 
    embedding_client: OpenAI, 
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 100
) -> List[List[float]]:
    """Get embeddings using an OpenAI-compatible embedding API."""
    all_embeddings: List[List[float]] = []
    
    print(f"  Computing embeddings in batches of {batch_size}...")
    print(f"  Using embedding model: {embedding_model}")
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        try:
            response = embedding_client.embeddings.create(
                model=embedding_model,
                input=batch,
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"  Error getting embeddings for batch {i}-{i+batch_size}: {e}")
            # Retry with smaller batch
            for text in batch:
                try:
                    response = embedding_client.embeddings.create(
                        model=embedding_model,
                        input=[text],
                    )
                    all_embeddings.append(response.data[0].embedding)
                except Exception as e2:
                    print(f"    Failed to get embedding for text: {e2}")
                    # Return zero vector as fallback
                    all_embeddings.append([0.0] * 1024)  # Default dimension
        
        print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
    
    return all_embeddings


def are_findings_equivalent(
    text1: str, 
    text2: str, 
    llm_client: OpenAI,
    llm_model: str = DEFAULT_LLM_MODEL,
    max_retries: int = 3
) -> bool:
    """
    Ask LLM if two findings are semantically equivalent.
    Uses a local OpenAI-compatible API.
    """
    prompt = f"""You are an expert researcher. Determine if the following two atomic findings are describing the same fact or finding.
They might be phrased differently, but if they convey the same core information, say YES.
If they are different, contradictory, or unrelated, say NO.

Finding 1: "{text1}"
Finding 2: "{text2}"

Answer only with YES or NO."""

    for attempt in range(max_retries):
        try:
            response = llm_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
                seed=42,  # Required by SGLang when temperature=0
            )
            answer = response.choices[0].message.content.strip().upper()
            return "YES" in answer
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))  # Exponential backoff
            else:
                print(f"LLM verification failed after {max_retries} retries: {e}")
                return False
    return False



def cluster_findings(
    embeddings: List[List[float]], 
    findings_texts: List[str],
    finding_groups: List[Optional[str]],
    llm_client: OpenAI,
    llm_model: str = DEFAULT_LLM_MODEL,
    threshold: float = 0.82,
    max_workers: int = 4
) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    """
    Cluster findings based on cosine similarity using union-find,
    verified by LLM for high-similarity pairs.
    Only compares findings within the same group.
    """
    n = len(embeddings)
    dsu = DisjointSet(n)
    
    # Group findings by their group identifier
    groups_dict: Dict[Optional[str], List[int]] = {}
    for idx, group_id in enumerate(finding_groups):
        if group_id not in groups_dict:
            groups_dict[group_id] = []
        groups_dict[group_id].append(idx)
    
    print(f"  Comparing {n} findings across {len(groups_dict)} group(s) (threshold={threshold})...")
    for group_id, group_indices in groups_dict.items():
        print(f"    Group '{group_id}': {len(group_indices)} findings")
    
    # Compare pairs ONLY within the same group
    llm_checks = 0
    llm_matches = 0
    
    # Collect all pairs that need LLM verification
    pairs_to_verify: List[Tuple[int, int]] = []
    
    for group_id, group_indices in groups_dict.items():
        for idx_i, i in enumerate(group_indices):
            for j in group_indices[idx_i + 1:]:
                similarity = cosine_similarity(embeddings[i], embeddings[j])
                if similarity >= threshold:
                    pairs_to_verify.append((i, j))
    
    llm_checks = len(pairs_to_verify)
    print(f"  Found {llm_checks} high-similarity pairs to verify with LLM...")
    
    # Verify pairs in parallel
    if pairs_to_verify:
        if max_workers > 1:
            print(f"  Verifying pairs in parallel with {max_workers} workers...")
            verified_pairs = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_pair = {
                    executor.submit(
                        are_findings_equivalent,
                        findings_texts[i],
                        findings_texts[j],
                        llm_client,
                        llm_model
                    ): (i, j)
                    for i, j in pairs_to_verify
                }
                
                for future in as_completed(future_to_pair):
                    i, j = future_to_pair[future]
                    try:
                        if future.result():
                            verified_pairs.append((i, j))
                            llm_matches += 1
                    except Exception as e:
                        print(f"    Warning: LLM verification failed for pair ({i}, {j}): {e}")
            
            # Apply unions for verified pairs
            for i, j in verified_pairs:
                dsu.union(i, j)
        else:
            # Sequential processing
            for i, j in pairs_to_verify:
                if are_findings_equivalent(findings_texts[i], findings_texts[j], llm_client, llm_model):
                    dsu.union(i, j)
                    llm_matches += 1
    
    print(f"  ✓ LLM Verification: {llm_matches}/{llm_checks} pairs confirmed (within-group only)")

    
    # Get clusters and assign sequential IDs
    raw_clusters = dsu.get_clusters()
    root_to_cluster_id = {root: idx for idx, root in enumerate(sorted(raw_clusters.keys()))}
    
    cluster_map: Dict[int, int] = {}
    for i in range(n):
        root = dsu.find(i)
        cluster_map[i] = root_to_cluster_id[root]
    
    clusters: Dict[int, List[int]] = {}
    for finding_idx, cluster_id in cluster_map.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(finding_idx)
    
    return cluster_map, clusters


def build_canonical_finding_space(
    atomic_facts_files: List[Dict[str, Any]], 
    threshold: float = 0.82,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
    embedding_base_url: Optional[str] = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    source_filenames: Optional[List[str]] = None,
    max_workers: int = 4
) -> Tuple[List[Set[int]], Dict[int, int], Dict[str, Any], List[str], List[Tuple[int, int]], List[Optional[str]], List[List[str]]]:
    """
    Build canonical finding space from multiple atomic_facts files.
    Only compares findings within the same group (based on filename pattern).
    
    Args:
        atomic_facts_files: List of atomic facts data (from JSON files)
        threshold: Cosine similarity threshold for clustering (default: 0.82)
        llm_base_url: Base URL for local LLM server
        llm_model: LLM model name for semantic verification
        embedding_base_url: Base URL for embedding server (defaults to Together API for embeddings)
        embedding_model: Embedding model name
        source_filenames: Optional list of source filenames (for explicit mapping and group extraction)
        max_workers: Maximum number of parallel workers for LLM verification (default: 4)
        
    Returns:
        - canonical_files: List of sets, where each set contains cluster IDs for a file
        - cluster_map: Mapping from global finding index to cluster ID
        - metadata: Dictionary with processing statistics
        - all_findings: List of all atomic finding strings
        - finding_origins: List of (file_idx, claim_idx) for each finding
        - finding_groups: List of group identifiers for each finding
        - finding_citations: List of citation lists for each finding
    """
    # Initialize LLM client
    llm_client = OpenAI(
        api_key=DEFAULT_LLM_API_KEY,
        base_url=llm_base_url,
    )
    print(f"  LLM client initialized: {llm_base_url}")
    print(f"  LLM model: {llm_model}")
    
    # Initialize embedding client (can be local or Together API)
    if embedding_base_url:
        embedding_client = OpenAI(
            api_key=DEFAULT_LLM_API_KEY,
            base_url=embedding_base_url,
        )
        print(f"  Embedding client: {embedding_base_url}")
    else:
        # Fall back to Together API for embeddings (requires TOGETHER_API_KEY)
        from together import Together
        together_client = Together()
        # Wrap Together client to match OpenAI interface
        class TogetherEmbeddingWrapper:
            def __init__(self, client):
                self.client = client
            
            def create(self, model, input):
                return self.client.embeddings.create(model=model, input=input)
        
        class EmbeddingClientWrapper:
            def __init__(self, together_client):
                self.embeddings = TogetherEmbeddingWrapper(together_client)
        
        embedding_client = EmbeddingClientWrapper(together_client)
        print(f"  Embedding client: Together API")
    
    print(f"  Embedding model: {embedding_model}")
    
    # Extract all atomic findings
    print(f"\n Extracting atomic findings from {len(atomic_facts_files)} file(s)...")
    all_findings, finding_origins, finding_groups, finding_citations = extract_all_atomic_findings(atomic_facts_files, source_filenames)
    print(f"  Total atomic findings: {len(all_findings)}")
    
    # Count unique groups
    unique_groups = set(finding_groups)
    print(f"  Detected groups: {sorted([g for g in unique_groups if g is not None])}")
    
    # Count citations
    total_citations = sum(1 for c in finding_citations if c)
    print(f"  Findings with citations: {total_citations}")
    
    # Get embeddings
    print(f"\n Computing embeddings...")
    embeddings = get_embeddings(all_findings, embedding_client, embedding_model)
    print(f"  ✓ Embeddings computed: {len(embeddings)}")
    
    # Cluster findings (within groups only)
    print(f"\n Clustering findings (within-group comparisons only)...")
    cluster_map, clusters = cluster_findings(
        embeddings, all_findings, finding_groups, 
        llm_client, llm_model, threshold, max_workers
    )

    print(f"  ✓ Total clusters: {len(clusters)}")
    
    # Build canonical representations for each file
    canonical_files: List[Set[int]] = []
    for file_idx in range(len(atomic_facts_files)):
        file_clusters: Set[int] = set()
        for finding_idx, (origin_file_idx, _) in enumerate(finding_origins):
            if origin_file_idx == file_idx:
                cluster_id = cluster_map[finding_idx]
                file_clusters.add(cluster_id)
        canonical_files.append(file_clusters)
    
    # Compute metadata (including group information)
    metadata: Dict[str, Any] = {
        'num_files': len(atomic_facts_files),
        'total_findings': len(all_findings),
        'total_clusters': len(clusters),
        'threshold': threshold,
        'llm_model': llm_model,
        'embedding_model': embedding_model,
        'comparison_mode': 'within_group_only',
        'unique_groups': sorted([g for g in unique_groups if g is not None]),
        'cluster_sizes': {
            cluster_id: len(members) 
            for cluster_id, members in clusters.items()
        },
        'findings_per_file': [
            len([1 for origin_idx, _ in finding_origins if origin_idx == file_idx])
            for file_idx in range(len(atomic_facts_files))
        ],
        'source_filenames': source_filenames if source_filenames else [],
        'findings_with_citations': total_citations
    }
    
    return canonical_files, cluster_map, metadata, all_findings, finding_origins, finding_groups, finding_citations


def save_results(
    canonical_files: List[Set[int]],
    cluster_map: Dict[int, int],
    metadata: Dict[str, Any],
    output_path: str,
    all_findings: List[str],
    finding_origins: List[Tuple[int, int]],
    finding_groups: List[Optional[str]],
    finding_citations: Optional[List[List[str]]] = None,
):
    """Save the canonical finding space results to a JSON file."""
    
    # Build an interpretable list of all findings with their cluster + origin + group + citations
    findings_detailed: List[Dict[str, Any]] = []
    for idx, (text, (file_idx, claim_idx), group_id) in enumerate(zip(all_findings, finding_origins, finding_groups)):
        finding_entry = {
            "finding_index": idx,
            "text": text,
            "cluster_id": cluster_map[idx],
            "file_index": file_idx,
            "claim_index": claim_idx,
            "group_id": group_id,
        }
        # Add citations if available
        if finding_citations and idx < len(finding_citations):
            finding_entry["citations"] = finding_citations[idx]
        findings_detailed.append(finding_entry)
    
    results = {
        'canonical_files': [list(clusters) for clusters in canonical_files],
        'cluster_map': cluster_map,
        'metadata': metadata,
        'findings': findings_detailed,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n Results saved to: {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Atomic Findings Embedding Pipeline - Extract and cluster semantic findings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all atomic_facts files in a directory
  python atomic_findings_pipeline.py /path/to/atomic_facts_dir/

  # Process specific files
  python atomic_findings_pipeline.py file1.json file2.json

  # Use custom threshold
  python atomic_findings_pipeline.py --threshold 0.85 /path/to/dir/

  # Specify output file
  python atomic_findings_pipeline.py --output results.json /path/to/dir/

  # Use custom LLM server
  python atomic_findings_pipeline.py --llm-base-url http://localhost:30000/v1 --llm-model Qwen/Qwen3-30B-A3B-Instruct-2507 /path/to/dir/
        """
    )
    
    parser.add_argument(
        'input',
        nargs='+',
        help='Path(s) to atomic_facts JSON file(s) or directory containing them'
    )
    
    parser.add_argument(
        '--threshold',
        '-t',
        type=float,
        default=0.82,
        help='Similarity threshold for clustering (default: 0.82)'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        default='canonical_findings.json',
        help='Output file path (default: canonical_findings.json)'
    )
    
    parser.add_argument(
        '--llm-base-url',
        type=str,
        default=DEFAULT_LLM_BASE_URL,
        help=f'Base URL for local LLM server (default: {DEFAULT_LLM_BASE_URL})'
    )
    
    parser.add_argument(
        '--llm-model',
        type=str,
        default=DEFAULT_LLM_MODEL,
        help=f'LLM model name for semantic verification (default: {DEFAULT_LLM_MODEL})'
    )
    
    parser.add_argument(
        '--embedding-base-url',
        type=str,
        default=None,
        help='Base URL for embedding server (default: use Together API)'
    )
    
    parser.add_argument(
        '--embedding-model',
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f'Embedding model name (default: {DEFAULT_EMBEDDING_MODEL})'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of parallel workers for LLM verification (default: 4, set to 1 for sequential)'
    )
    
    args = parser.parse_args()
    
    # Find all atomic_facts files
    file_paths: List[Path] = []
    for input_path in args.input:
        path = Path(input_path)
        if path.is_file() and path.suffix == '.json':
            file_paths.append(path)
        elif path.is_dir():
            file_paths.extend(sorted(path.rglob("atomic_facts_*.json")))
        else:
            print(f"  Skipping invalid path: {input_path}")
    
    if not file_paths:
        print(" No atomic_facts JSON files found")
        return 1
    
    print("="*80)
    print("ATOMIC FINDINGS EMBEDDING PIPELINE")
    print("="*80)
    print(f"\n Found {len(file_paths)} atomic_facts file(s):")
    for fp in file_paths[:5]:
        print(f"   • {fp.name}")
    if len(file_paths) > 5:
        print(f"   ... and {len(file_paths) - 5} more")
    
    # Load files
    print(f"\n Loading atomic_facts files...")
    atomic_facts_files: List[Dict[str, Any]] = []
    loaded_filenames: List[str] = []
    for filepath in file_paths:
        try:
            data = load_atomic_facts_file(str(filepath))
            atomic_facts_files.append(data)
            loaded_filenames.append(filepath.name)
            print(f"   ✓ {filepath.name}")
        except Exception as e:
            print(f"    Error loading {filepath.name}: {e}")
    
    if not atomic_facts_files:
        print("\n No files loaded successfully")
        return 1
    
    print(f"\n Successfully loaded {len(atomic_facts_files)} file(s)")
    print(f" Using threshold: {args.threshold}")
    print(f" Max workers: {args.max_workers}")
    print(f" LLM base URL: {args.llm_base_url}")
    print(f" LLM model: {args.llm_model}")
    
    # Build canonical finding space
    try:
        (
            canonical_files,
            cluster_map,
            metadata,
            all_findings,
            finding_origins,
            finding_groups,
            finding_citations,
        ) = build_canonical_finding_space(
            atomic_facts_files=atomic_facts_files,
            threshold=args.threshold,
            llm_base_url=args.llm_base_url,
            llm_model=args.llm_model,
            embedding_base_url=args.embedding_base_url,
            embedding_model=args.embedding_model,
            source_filenames=loaded_filenames,
            max_workers=args.max_workers
        )
        
        # Display summary
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        
        print(f"\n Overall Statistics:")
        print(f"   • Total files: {metadata['num_files']}")
        print(f"   • Total atomic findings: {metadata['total_findings']}")
        print(f"   • Total canonical clusters: {metadata['total_clusters']}")
        print(f"   • Compression ratio: {metadata['total_findings'] / metadata['total_clusters']:.2f}x")
        print(f"   • Comparison mode: {metadata['comparison_mode']}")
        print(f"   • Groups detected: {metadata['unique_groups']}")
        print(f"   • Findings with citations: {metadata['findings_with_citations']}")
        
        print(f"\n Per-File Canonical Findings:")
        for idx, clusters in enumerate(canonical_files):
            num_findings = metadata['findings_per_file'][idx]
            filename = loaded_filenames[idx] if idx < len(loaded_filenames) else f"File {idx}"
            print(f"   {filename}: {len(clusters)} unique clusters (from {num_findings} findings)")
        
        # Save results
        save_results(
            canonical_files=canonical_files,
            cluster_map=cluster_map,
            metadata=metadata,
            output_path=args.output,
            all_findings=all_findings,
            finding_origins=finding_origins,
            finding_groups=finding_groups,
            finding_citations=finding_citations,
        )
        
        print("\n Pipeline complete!")
        return 0
        
    except Exception as e:
        print(f"\n Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
