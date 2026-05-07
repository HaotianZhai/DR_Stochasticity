#!/bin/bash
###############################################################################
# Full Stochasticity Evaluation Pipeline
#
# This script processes all trajectories and:
# 1. Extracts QA answers from trajectory reports
# 2. Extracts claims and decomposes them into atomic facts
# 3. Clusters atomic findings semantically
# 4. Calculates stochasticity metrics
#
# Usage:
#   ./run_full_stochasticity_pipeline.sh --input-dir /path/to/trajectories [OPTIONS]
#
# Requirements:
#   - TOGETHER_API_KEY environment variable set (for embeddings)
#   - Local LLM server running (SGLang) or Together API
###############################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:30000/v1}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-intfloat/multilingual-e5-large-instruct}"

OUTPUT_DIR="stochasticity_results_$(date +%Y%m%d_%H%M%S)"
TRAJECTORIES_DIR=""
CLUSTERING_THRESHOLD=0.94
TEMPERATURE=0.1
MAX_WORKERS=10
USE_LLM=true
CALCULATE_ACCURACY=false
REFERENCE_FILE=""

CLAIM_EXTRACTION_TOOL="${SCRIPT_DIR}/../claim_extraction/extract_claims.py"
CLUSTERING_TOOL="${SCRIPT_DIR}/../atomic_findings/atomic_findings_pipeline.py"
STOCHASTICITY_TOOL="${SCRIPT_DIR}/../atomic_findings/calculate_stochasticity.py"

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --input-dir)        TRAJECTORIES_DIR="$2"; shift 2 ;;
        --threshold)        CLUSTERING_THRESHOLD="$2"; shift 2 ;;
        --temperature)      TEMPERATURE="$2"; shift 2 ;;
        --max-workers)      MAX_WORKERS="$2"; shift 2 ;;
        --no-llm)           USE_LLM=false; shift ;;
        --llm-base-url)     LLM_BASE_URL="$2"; shift 2 ;;
        --llm-model)        LLM_MODEL="$2"; shift 2 ;;
        --embedding-model)  EMBEDDING_MODEL="$2"; shift 2 ;;
        --accuracy)         CALCULATE_ACCURACY=true; shift ;;
        --reference)        REFERENCE_FILE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --input-dir <trajectories_dir> [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --output-dir DIR           Output directory (default: stochasticity_results_TIMESTAMP)"
            echo "  --input-dir DIR            Trajectories directory (required)"
            echo "  --threshold FLOAT          Clustering similarity threshold (default: 0.94)"
            echo "  --temperature FLOAT        LLM temperature for extraction (default: 0.1)"
            echo "  --max-workers INT          Number of parallel workers (default: 10)"
            echo "  --no-llm                   Disable LLM semantic comparison"
            echo "  --llm-base-url URL         Base URL for local LLM server"
            echo "  --llm-model NAME           LLM model name"
            echo "  --embedding-model NAME     Embedding model name (via Together API)"
            echo "  --accuracy                 Calculate accuracy (requires --reference)"
            echo "  --reference FILE           Reference QA pairs JSON file"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            exit 1
            ;;
    esac
done

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║        FULL STOCHASTICITY EVALUATION PIPELINE                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

if [ -z "$TOGETHER_API_KEY" ]; then
    echo -e "${RED}Error: TOGETHER_API_KEY environment variable not set${NC}"
    echo "Required for embedding computation. Set with: export TOGETHER_API_KEY='your-key'"
    exit 1
fi

if [ -z "$TRAJECTORIES_DIR" ]; then
    echo -e "${RED}Error: --input-dir is required${NC}"
    exit 1
fi

if [ ! -d "$TRAJECTORIES_DIR" ]; then
    echo -e "${RED}Error: Trajectories directory not found: $TRAJECTORIES_DIR${NC}"
    exit 1
fi

NUM_TRAJECTORIES=$(ls -1 "$TRAJECTORIES_DIR"/deep_research_trajectory_*.json 2>/dev/null | wc -l | tr -d ' ')
if [ "$NUM_TRAJECTORIES" -eq 0 ]; then
    echo -e "${RED}Error: No trajectory files found in $TRAJECTORIES_DIR${NC}"
    exit 1
fi

echo -e "${BLUE}Configuration:${NC}"
echo "  Output directory:      $OUTPUT_DIR"
echo "  Trajectories:          $NUM_TRAJECTORIES files"
echo "  Clustering threshold:  $CLUSTERING_THRESHOLD"
echo "  Temperature:           $TEMPERATURE"
echo "  Max workers:           $MAX_WORKERS"
echo "  LLM base URL:          $LLM_BASE_URL"
echo "  LLM model:             $LLM_MODEL"
echo "  Embedding model:       $EMBEDDING_MODEL (via Together API)"
echo "  Use LLM comparison:    $USE_LLM"
echo "  Accuracy:              $CALCULATE_ACCURACY"
if [ "$CALCULATE_ACCURACY" = true ]; then
    echo "  Reference file:        $REFERENCE_FILE"
fi
echo ""

mkdir -p "$OUTPUT_DIR"/{answers,claims,atomic_facts,clustered,metrics}

START_TIME=$(date +%s)

###############################################################################
# STEP 1: Extract QA Answers
###############################################################################

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}STEP 1: Extracting QA Answers from Trajectories${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

STEP1_START=$(date +%s)

python "$CLAIM_EXTRACTION_TOOL" \
    --input-dir "$TRAJECTORIES_DIR" \
    --output-dir "$OUTPUT_DIR/answers" \
    --mode qa_answers \
    --llm-base-url "$LLM_BASE_URL" \
    --model "$LLM_MODEL" \
    --temperature "$TEMPERATURE" \
    --max-workers "$MAX_WORKERS" \
    2>&1 | tee "$OUTPUT_DIR/step1_extract_answers.log"

NUM_ANSWERS=$(ls -1 "$OUTPUT_DIR/answers"/qa_answer_*.json 2>/dev/null | wc -l | tr -d ' ')
STEP1_END=$(date +%s)
STEP1_DURATION=$((STEP1_END - STEP1_START))

echo -e "${GREEN}✓ Step 1 Complete: $NUM_ANSWERS answers extracted (${STEP1_DURATION}s)${NC}"
echo ""

###############################################################################
# STEP 2: Extract Claims & Decompose into Atomic Facts
###############################################################################

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}STEP 2: Extracting Claims & Decomposing into Atomic Facts${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

STEP2_START=$(date +%s)

python "$CLAIM_EXTRACTION_TOOL" \
    --input-dir "$TRAJECTORIES_DIR" \
    --output-dir "$OUTPUT_DIR/claims" \
    --mode claims \
    --llm-base-url "$LLM_BASE_URL" \
    --model "$LLM_MODEL" \
    --temperature "$TEMPERATURE" \
    --max-workers "$MAX_WORKERS" \
    2>&1 | tee "$OUTPUT_DIR/step2_extract_claims.log"

echo ""
echo "Decomposing into atomic facts..."
python "$CLAIM_EXTRACTION_TOOL" \
    --input-dir "$OUTPUT_DIR/claims" \
    --output-dir "$OUTPUT_DIR/atomic_facts" \
    --mode atomic_facts \
    --llm-base-url "$LLM_BASE_URL" \
    --model "$LLM_MODEL" \
    --temperature "$TEMPERATURE" \
    --max-workers "$MAX_WORKERS" \
    2>&1 | tee "$OUTPUT_DIR/step2_extract_atomic_facts.log"

NUM_ATOMIC_FACTS=$(ls -1 "$OUTPUT_DIR/atomic_facts"/atomic_facts_*.json 2>/dev/null | wc -l | tr -d ' ')
STEP2_END=$(date +%s)
STEP2_DURATION=$((STEP2_END - STEP2_START))

echo -e "${GREEN}✓ Step 2 Complete: $NUM_ATOMIC_FACTS atomic facts files (${STEP2_DURATION}s)${NC}"
echo ""

###############################################################################
# STEP 3: Cluster Atomic Findings
###############################################################################

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}STEP 3: Clustering Atomic Findings${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

STEP3_START=$(date +%s)

python "$CLUSTERING_TOOL" \
    "$OUTPUT_DIR/atomic_facts"/atomic_facts_*.json \
    --threshold "$CLUSTERING_THRESHOLD" \
    --output "$OUTPUT_DIR/clustered/clustered_findings.json" \
    --llm-base-url "$LLM_BASE_URL" \
    --llm-model "$LLM_MODEL" \
    --embedding-model "$EMBEDDING_MODEL" \
    --max-workers "$MAX_WORKERS" \
    2>&1 | tee "$OUTPUT_DIR/step3_clustering.log"

STEP3_END=$(date +%s)
STEP3_DURATION=$((STEP3_END - STEP3_START))

if [ -f "$OUTPUT_DIR/clustered/clustered_findings.json" ]; then
    TOTAL_FINDINGS=$(python3 -c "import json; d=json.load(open('$OUTPUT_DIR/clustered/clustered_findings.json')); print(d['metadata']['total_findings'])")
    TOTAL_CLUSTERS=$(python3 -c "import json; d=json.load(open('$OUTPUT_DIR/clustered/clustered_findings.json')); print(d['metadata']['total_clusters'])")
    echo -e "${GREEN}✓ Step 3 Complete: $TOTAL_FINDINGS findings → $TOTAL_CLUSTERS clusters (${STEP3_DURATION}s)${NC}"
else
    echo -e "${RED}✗ Step 3 Failed: clustering output not found${NC}"
    exit 1
fi
echo ""

###############################################################################
# STEP 4: Calculate Stochasticity Metrics
###############################################################################

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}STEP 4: Calculating Stochasticity Metrics${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

STEP4_START=$(date +%s)

STOCH_CMD="python $STOCHASTICITY_TOOL \
    $OUTPUT_DIR/clustered/clustered_findings.json \
    $OUTPUT_DIR/answers \
    --llm-base-url $LLM_BASE_URL \
    --model $LLM_MODEL \
    --max-workers $MAX_WORKERS \
    --output $OUTPUT_DIR/metrics/stochasticity_metrics.json"

if [ "$USE_LLM" = false ]; then
    STOCH_CMD="$STOCH_CMD --no-llm"
fi

if [ "$CALCULATE_ACCURACY" = true ] && [ -n "$REFERENCE_FILE" ]; then
    STOCH_CMD="$STOCH_CMD --accuracy --reference $REFERENCE_FILE"
fi

eval "$STOCH_CMD" 2>&1 | tee "$OUTPUT_DIR/step4_stochasticity.log"

STEP4_END=$(date +%s)
STEP4_DURATION=$((STEP4_END - STEP4_START))

echo -e "${GREEN}✓ Step 4 Complete (${STEP4_DURATION}s)${NC}"
echo ""

###############################################################################
# Summary
###############################################################################

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}PIPELINE COMPLETE ✓${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Timing:"
echo "  Step 1 (Extract answers):     ${STEP1_DURATION}s"
echo "  Step 2 (Atomic facts):        ${STEP2_DURATION}s"
echo "  Step 3 (Clustering):          ${STEP3_DURATION}s"
echo "  Step 4 (Stochasticity):       ${STEP4_DURATION}s"
echo "  Total time:                   ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "  ├── answers/                  QA answer files"
echo "  ├── claims/                   Extracted claims"
echo "  ├── atomic_facts/             Atomic facts files"
echo "  ├── clustered/                Clustered findings"
echo "  └── metrics/                  Stochasticity metrics"
echo ""

if [ -f "$OUTPUT_DIR/metrics/stochasticity_metrics.json" ]; then
    echo -e "${BLUE}Final Stochasticity Metrics:${NC}"
    python3 -c "import json; print(json.dumps(json.load(open('$OUTPUT_DIR/metrics/stochasticity_metrics.json')), indent=2))" 2>/dev/null
    echo ""
fi

echo -e "${GREEN}All done! Results are in: $OUTPUT_DIR${NC}"
